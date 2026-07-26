from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from smart_reward import phase2_r3_terminal as terminal
from smart_reward.phase2_r3_identity import (
    CONTINUABLE_PRIMARY_TERMINAL_ROLE,
    CONTINUABLE_PRIMARY_TERMINAL_SCHEMA,
    R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA,
    SUCCESSFUL_PROFILE_TERMINAL_ROLE,
    SUCCESSFUL_PROFILE_TERMINAL_SCHEMA,
    VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
    VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
)
from smart_reward.phase2_r3_orchestrator import (
    SEGMENT_OUTCOME_SCHEMA,
    primary_outcome_semantic_sha256,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw_row(
    *,
    job_id: str = "410000_0",
    job_id_raw: str = "410001",
    state: str = "COMPLETED",
    exit_code: str = "0:0",
    derived_exit_code: str = "0:0",
    cluster: str = "hpc4",
    account: str = "sigroup",
    partition: str = "gpu-l20",
    qos: str = "l20_qos",
    n_nodes: str = "1",
    n_cpus: str = "4",
    req_tres: str = "billing=4,cpu=4,gres/gpu=1,mem=2G,node=1",
    alloc_tres: str = ("billing=4,cpu=4,gres/gpu=1,mem=2G,node=1,gres/gpu:l20=1"),
    elapsed_raw: str = "137",
) -> bytes:
    return (
        "|".join(
            (
                job_id,
                job_id_raw,
                state,
                exit_code,
                derived_exit_code,
                cluster,
                account,
                partition,
                qos,
                n_nodes,
                n_cpus,
                req_tres,
                alloc_tres,
                elapsed_raw,
            )
        )
        + "\n"
    ).encode()


def _inspection(**changes: str) -> terminal.ClaimFreeSacctTerminalInspection:
    raw = _raw_row(**changes)
    return terminal.inspect_sacct_terminal_bytes(
        raw,
        expected_raw_sha256=_sha(raw),
    )


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        partition="gpu-l20",
        gpu_name="NVIDIA L20",
        gpus_per_task=1,
        cpus_per_task=4,
        memory_bytes=2 * 1024**3,
        requested_walltime_seconds_per_segment=3600,
        resource_plan_sha256="5" * 64,
        file_sha256="6" * 64,
        size_bytes=1024,
        bundle_semantic_sha256="7" * 64,
        profile_run_sha256="1" * 64,
        formal_profile_sha256="3" * 64,
        max_scheduler_segments=3,
    )


def _accept_fake_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        terminal,
        "_validated_operational_bundle",
        lambda value: value,
    )


def _expected_resources(
    *,
    cpus_per_task: int = 4,
    memory_bytes: int = 2 * 1024**3,
    requested_walltime_seconds: int = 3600,
) -> dict[str, object]:
    return {
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "gpu-l20",
        "gpu_name": "NVIDIA L20",
        "slurm_gpu_tres": "gres/gpu:l20",
        "gpus_per_task": 1,
        "cpus_per_task": cpus_per_task,
        "memory_bytes": memory_bytes,
        "nodes": 1,
        "requested_walltime_seconds": requested_walltime_seconds,
    }


def test_claim_free_inspection_binds_exact_raw_bytes_but_does_not_promote() -> None:
    inspection = _inspection(state="TIMEOUT", exit_code="0:15")

    assert inspection.to_dict()["formal_claim_eligible"] is False
    assert inspection.row.state == "TIMEOUT"
    assert inspection.raw_sacct_sha256 == _sha(inspection.raw_bytes)
    with pytest.raises(ValueError, match="COMPLETED"):
        terminal._validate_terminal_row(
            inspection,
            expected_job_id="410000_0",
            expected_job_id_raw="410001",
            expected_resources=_expected_resources(),
            requested_walltime_seconds=3600,
        )


def test_inspection_requires_caller_digest_and_one_exact_locked_row(tmp_path: Path) -> None:
    raw = _raw_row()
    with pytest.raises(ValueError, match="expected_raw_sha256"):
        terminal.inspect_sacct_terminal_bytes(
            raw,
            expected_raw_sha256="0" * 64,
        )
    for invalid in (
        raw.rstrip(b"\n"),
        raw + raw,
        raw.replace(b"|137\n", b"|137|extra\n"),
        raw.replace(
            b"billing=4,cpu=4",
            b"billing=4,billing=4,cpu=4",
            1,
        ),
    ):
        with pytest.raises(ValueError):
            terminal.inspect_sacct_terminal_bytes(
                invalid,
                expected_raw_sha256=_sha(invalid),
            )

    path = (tmp_path / "caller.sacct.psv").resolve()
    path.write_bytes(raw)
    loaded = terminal.inspect_sacct_terminal_file(
        path,
        expected_raw_sha256=_sha(raw),
    )
    assert loaded.raw_bytes == raw


@pytest.mark.parametrize("state", ["TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"])
def test_terminal_validator_rejects_non_success_states(
    state: str,
) -> None:
    with pytest.raises(ValueError, match="COMPLETED"):
        terminal._validate_terminal_row(
            _inspection(state=state),
            expected_job_id="410000_0",
            expected_job_id_raw="410001",
            expected_resources=_expected_resources(),
            requested_walltime_seconds=3600,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"exit_code": "1:0"}, "both exits"),
        ({"derived_exit_code": "1:0"}, "both exits"),
        ({"cluster": "other"}, "HPC4 allocation"),
        ({"account": "other"}, "HPC4 allocation"),
        ({"partition": "gpu-a100"}, "HPC4 allocation"),
        ({"n_nodes": "2"}, "HPC4 allocation"),
        ({"n_cpus": "8"}, "HPC4 allocation"),
        ({"elapsed_raw": "0"}, "elapsed"),
        ({"elapsed_raw": "3601"}, "elapsed"),
        ({"req_tres": "billing=4,cpu=4,gres/gpu=1,mem=4G,node=1"}, "TRES"),
        (
            {"alloc_tres": ("billing=4,cpu=4,gres/gpu=1,mem=2G,node=1,gres/gpu:a100=1")},
            "TRES",
        ),
    ],
)
def test_terminal_validator_rejects_identity_resource_and_elapsed_drift(
    changes: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        terminal._validate_terminal_row(
            _inspection(**changes),
            expected_job_id="410000_0",
            expected_job_id_raw="410001",
            expected_resources=_expected_resources(),
            requested_walltime_seconds=3600,
        )


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ("410000", "410000_0"),
        ("410000_0.batch", "410000_0.batch"),
        ("410000_[0-2]", "410000_[0-2]"),
    ],
)
def test_terminal_validator_rejects_parent_step_and_range_rows(
    observed: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match="job/task|parent"):
        terminal._validate_terminal_row(
            _inspection(job_id=observed),
            expected_job_id=expected,
            expected_job_id_raw="410001",
            expected_resources=_expected_resources(),
            requested_walltime_seconds=3600,
        )


def _profile_fakes() -> tuple[SimpleNamespace, SimpleNamespace]:
    bundle = _bundle()
    intent = SimpleNamespace(
        file_sha256="a" * 64,
        allocation_intent_sha256="b" * 64,
        expected_slurm_resources=lambda: _expected_resources(),
    )
    receipt = SimpleNamespace(
        operational_bundle=bundle,
        allocation_intent=intent,
        sacct_job_selector="410000_0",
        job_id="410001",
        file_sha256="8" * 64,
        runtime_receipt_sha256="4" * 64,
        requested_walltime_seconds=3600,
    )
    return bundle, receipt


def test_profile_producer_publishes_reopens_and_replace_loses_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, receipt = _profile_fakes()
    _accept_fake_bundle(monkeypatch)
    monkeypatch.setattr(
        terminal,
        "_validate_profile_dependencies",
        lambda operational_bundle, *, runtime_receipt: (
            operational_bundle,
            runtime_receipt,
        ),
    )
    inspection = _inspection()
    directory = (tmp_path / "profile-terminal").resolve()

    capability = terminal.produce_successful_profile_terminal(
        bundle,  # type: ignore[arg-type]
        runtime_receipt=receipt,  # type: ignore[arg-type]
        inspection=inspection,
        evidence_directory=directory,
    )

    assert set(path.name for path in directory.iterdir()) == {
        "raw-sacct.psv",
        "parsed-sacct.json",
        "terminal-manifest.json",
    }
    assert (directory / "raw-sacct.psv").read_bytes() == inspection.raw_bytes
    ref = capability.artifact_ref()
    assert ref.schema_version == SUCCESSFUL_PROFILE_TERMINAL_SCHEMA
    assert ref.role == SUCCESSFUL_PROFILE_TERMINAL_ROLE
    assert ref.artifact_sha256 == capability.terminal_sha256

    reopened = terminal.revalidate_successful_profile_terminal(
        bundle,  # type: ignore[arg-type]
        runtime_receipt=receipt,  # type: ignore[arg-type]
        evidence_directory=directory,
        expected_manifest_file_sha256=capability.manifest_file_sha256,
        expected_raw_sacct_sha256=inspection.raw_sacct_sha256,
    )
    assert reopened.to_dict() == capability.to_dict()

    copied = replace(capability)
    with pytest.raises(TypeError, match="private validator"):
        copied.validate_integrity()
    with pytest.raises(FileExistsError, match="overwrite"):
        terminal.produce_successful_profile_terminal(
            bundle,  # type: ignore[arg-type]
            runtime_receipt=receipt,  # type: ignore[arg-type]
            inspection=inspection,
            evidence_directory=directory,
        )
    with pytest.raises(ValueError):
        terminal.revalidate_successful_profile_terminal(
            bundle,  # type: ignore[arg-type]
            runtime_receipt=receipt,  # type: ignore[arg-type]
            evidence_directory=directory,
            expected_manifest_file_sha256=capability.manifest_file_sha256,
            expected_raw_sacct_sha256="0" * 64,
        )


def _primary_fake_payloads(
    *,
    status: str,
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, dict[str, object]]:
    head_ids = ("1" * 64, "2" * 64)
    admission_unsigned: dict[str, object] = {
        "schema_version": R3_PRIMARY_SEGMENT_ADMISSION_SCHEMA,
        "design_sha256": "b" * 64,
        "materialization_attestation_sha256": "3" * 64,
        "task_id": 0,
        "seed": 20260801,
        "segment_index": 1,
        "logical_run_id": "d" * 64,
        "head_runs": [
            {"head": "bt_mle", "head_run_id": head_ids[0]},
            {"head": "prorm_plus", "head_run_id": head_ids[1]},
        ],
        "scheduler_segment_id": "e" * 64,
        "start_mode": "fresh_zero_head_fresh_adamw",
        "continuation_evidence_sha256": None,
    }
    admission_payload = {
        **admission_unsigned,
        "admission_sha256": terminal._canonical_sha256(admission_unsigned),
    }
    runtime_unsigned: dict[str, object] = {
        "schema_version": "phase2-recovery-r3-slurm-segment-runtime/v1",
        "design_sha256": admission_payload["design_sha256"],
        "admission_sha256": admission_payload["admission_sha256"],
        "scheduler_segment_id": admission_payload["scheduler_segment_id"],
        "segment_index": 1,
        "task_id": 0,
        "seed": 20260801,
        "cluster": "hpc4",
        "job_id": "410001",
        "array_job_id": "410000",
        "array_task_id": 0,
        "account": "sigroup",
        "partition": "gpu-l20",
        "requested_walltime_seconds": 3600,
        "captured_monotonic_ns": 123,
    }
    runtime_payload = {
        **runtime_unsigned,
        "runtime_sha256": terminal._canonical_sha256(runtime_unsigned),
    }
    continuation = status == "continuation_required_after_safe_checkpoint"
    checkpoint = (
        {
            "schema_version": VERIFIED_CONTINUATION_CHECKPOINT_SCHEMA,
            "artifact_sha256": "a" * 64,
            "role": VERIFIED_CONTINUATION_CHECKPOINT_ROLE,
        }
        if continuation
        else None
    )
    completed = (
        []
        if continuation
        else [
            {
                "learner": "bt_mle",
                "head_run_id": head_ids[0],
                "completion_receipt_sha256": "4" * 64,
            },
            {
                "learner": "prorm_plus",
                "head_run_id": head_ids[1],
                "completion_receipt_sha256": "8" * 64,
            },
        ]
    )
    outcome_unsigned: dict[str, object] = {
        "schema_version": SEGMENT_OUTCOME_SCHEMA,
        "status": status,
        "design_sha256": admission_payload["design_sha256"],
        "admission_sha256": admission_payload["admission_sha256"],
        "logical_run_id": admission_payload["logical_run_id"],
        "scheduler_segment_id": admission_payload["scheduler_segment_id"],
        "runtime_sha256": runtime_payload["runtime_sha256"],
        "segment_index": 1,
        "task_id": 0,
        "seed": 20260801,
        "gate_p_resource_plan_sha256": "5" * 64,
        "completed_heads": completed,
        "active_learner": "bt_mle" if continuation else None,
        "continuation_checkpoint": checkpoint,
        "continuation_reason": "safe scheduler boundary" if continuation else None,
        "all_primary_heads_compute_complete": not continuation,
        "external_scheduler_terminal_required": True,
        "external_scheduler_success_claimed": False,
        "r3_success_authorization_created": False,
        "information_boundary": "train_only_head_free_segment_outcome",
    }
    outcome_payload = {
        **outcome_unsigned,
        "outcome_sha256": primary_outcome_semantic_sha256(outcome_unsigned),
    }
    return (
        SimpleNamespace(to_dict=lambda: admission_payload),
        SimpleNamespace(to_dict=lambda: runtime_payload),
        SimpleNamespace(),
        outcome_payload,
    )


def _publish_fake_primary_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
    filename: str,
) -> tuple[SimpleNamespace, terminal.PrimarySegmentRuntimeClosure]:
    bundle = _bundle()
    admission, runtime, outcome, outcome_payload = _primary_fake_payloads(status=status)
    _accept_fake_bundle(monkeypatch)
    monkeypatch.setattr(
        terminal,
        "_validate_primary_dependencies",
        lambda admitted, *, runtime, outcome, operational_bundle: (
            admitted,
            runtime,
            outcome,
            operational_bundle,
            outcome_payload,
        ),
    )
    attempt = (tmp_path / Path(filename).stem).resolve()
    (attempt / "runtime-closures").mkdir(parents=True)
    (attempt / "terminal-evidence").mkdir()
    closure = terminal.publish_primary_segment_runtime_closure(
        (attempt / "runtime-closures" / "task-0.json").resolve(),
        admission=admission,  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        operational_bundle=bundle,  # type: ignore[arg-type]
    )
    return bundle, closure


def test_primary_outcome_semantic_hash_is_newline_canonical_and_not_file_hash() -> None:
    admission, runtime, _, outcome = _primary_fake_payloads(
        status="continuation_required_after_safe_checkpoint"
    )
    unsigned = dict(outcome)
    outcome_sha = unsigned.pop("outcome_sha256")
    semantic_bytes = terminal.canonical_json_bytes(unsigned)

    assert semantic_bytes.endswith(b"\n")
    assert not semantic_bytes.endswith(b"\n\n")
    assert outcome_sha == hashlib.sha256(semantic_bytes).hexdigest()
    assert outcome_sha == primary_outcome_semantic_sha256(unsigned)
    assert outcome_sha != terminal._canonical_sha256(unsigned)
    assert outcome_sha != hashlib.sha256(terminal.canonical_json_bytes(outcome)).hexdigest()

    bundle = _bundle()
    terminal._validate_primary_closure_outcome(
        outcome,
        admission=admission.to_dict(),
        runtime=runtime.to_dict(),
        heads=admission.to_dict()["head_runs"],  # type: ignore[arg-type]
        operational_bundle=bundle,  # type: ignore[arg-type]
    )
    tampered = dict(outcome)
    tampered["continuation_reason"] = "tampered after hashing"
    with pytest.raises(ValueError, match="self-hash"):
        terminal._validate_primary_closure_outcome(
            tampered,
            admission=admission.to_dict(),
            runtime=runtime.to_dict(),
            heads=admission.to_dict()["head_runs"],  # type: ignore[arg-type]
            operational_bundle=bundle,  # type: ignore[arg-type]
        )

    wrong_format = dict(outcome)
    wrong_unsigned = dict(wrong_format)
    wrong_unsigned.pop("outcome_sha256")
    wrong_format["outcome_sha256"] = terminal._canonical_sha256(wrong_unsigned)
    with pytest.raises(ValueError, match="self-hash"):
        terminal._validate_primary_closure_outcome(
            wrong_format,
            admission=admission.to_dict(),
            runtime=runtime.to_dict(),
            heads=admission.to_dict()["head_runs"],  # type: ignore[arg-type]
            operational_bundle=bundle,  # type: ignore[arg-type]
        )


def test_primary_closures_split_continuation_and_completion_and_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, continuable_closure = _publish_fake_primary_closure(
        tmp_path,
        monkeypatch,
        status="continuation_required_after_safe_checkpoint",
        filename="continuable-closure.json",
    )
    bundle, completed_closure = _publish_fake_primary_closure(
        tmp_path,
        monkeypatch,
        status="compute_complete_pending_external_scheduler_terminal",
        filename="completed-closure.json",
    )
    assert continuable_closure.continuation_required is True
    assert completed_closure.continuation_required is False
    assert (
        completed_closure.to_dict()["producer_schema"]
        == terminal.PRIMARY_SEGMENT_RUNTIME_CLOSURE_PRODUCER_SCHEMA
    )
    reopened = terminal.reopen_primary_segment_runtime_closure(
        completed_closure.artifact_path,
        expected_file_sha256=completed_closure.file_sha256,
        operational_bundle=bundle,  # type: ignore[arg-type]
    )
    assert reopened.to_dict() == completed_closure.to_dict()
    with pytest.raises(TypeError, match="publish/reopen"):
        replace(completed_closure).validate_integrity()
    admission, runtime, outcome, _ = _primary_fake_payloads(
        status="compute_complete_pending_external_scheduler_terminal"
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        terminal.publish_primary_segment_runtime_closure(
            completed_closure.artifact_path,
            admission=admission,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            outcome=outcome,  # type: ignore[arg-type]
            operational_bundle=bundle,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="expected digest"):
        terminal.reopen_primary_segment_runtime_closure(
            completed_closure.artifact_path,
            expected_file_sha256="0" * 64,
            operational_bundle=bundle,  # type: ignore[arg-type]
        )
    tampered_payload = completed_closure.to_dict()
    tampered_payload["producer_schema"] = "attacker-rehashed/v1"
    tampered_unsigned = dict(tampered_payload)
    tampered_unsigned.pop("closure_sha256")
    tampered_payload["closure_sha256"] = terminal._canonical_sha256(tampered_unsigned)
    tampered = terminal.publish_canonical_artifact(
        (tmp_path / "tampered-closure.json").resolve(),
        tampered_payload,
    )
    with pytest.raises(ValueError, match="authority boundary"):
        terminal.reopen_primary_segment_runtime_closure(
            tampered.artifact_path,
            expected_file_sha256=tampered.file_sha256,
            operational_bundle=bundle,  # type: ignore[arg-type]
        )


def test_primary_terminal_capabilities_are_state_separated_and_post_job_pure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, continuable_closure = _publish_fake_primary_closure(
        tmp_path,
        monkeypatch,
        status="continuation_required_after_safe_checkpoint",
        filename="continuable-closure.json",
    )
    bundle, completed_closure = _publish_fake_primary_closure(
        tmp_path,
        monkeypatch,
        status="compute_complete_pending_external_scheduler_terminal",
        filename="completed-closure.json",
    )
    inspection = _inspection()
    continuable = terminal.produce_continuable_primary_terminal(
        bundle,  # type: ignore[arg-type]
        runtime_closure=continuable_closure,
        inspection=inspection,
        evidence_directory=(
            continuable_closure.artifact_path.parent.parent
            / "terminal-evidence"
            / "task-0-segment-1"
        ),
    )
    completed = terminal.produce_completed_primary_terminal(
        bundle,  # type: ignore[arg-type]
        runtime_closure=completed_closure,
        inspection=inspection,
        evidence_directory=(
            completed_closure.artifact_path.parent.parent / "terminal-evidence" / "task-0-segment-1"
        ),
    )
    assert continuable.artifact_ref().schema_version == CONTINUABLE_PRIMARY_TERMINAL_SCHEMA
    assert continuable.artifact_ref().role == CONTINUABLE_PRIMARY_TERMINAL_ROLE
    assert completed.artifact_ref().schema_version == terminal.COMPLETED_PRIMARY_TERMINAL_SCHEMA
    assert completed.artifact_ref().role == terminal.COMPLETED_PRIMARY_TERMINAL_ROLE
    assert completed.to_dict()["final_three_seed_authorization_issued"] is False
    assert completed.to_dict()["continuation_required"] is False
    assert continuable.to_dict()["continuation_required"] is True
    with pytest.raises(ValueError, match="status"):
        terminal.produce_completed_primary_terminal(
            bundle,  # type: ignore[arg-type]
            runtime_closure=continuable_closure,
            inspection=inspection,
            evidence_directory=(tmp_path / "wrong-completed").resolve(),
        )
    with pytest.raises(ValueError, match="status"):
        terminal.produce_continuable_primary_terminal(
            bundle,  # type: ignore[arg-type]
            runtime_closure=completed_closure,
            inspection=inspection,
            evidence_directory=(tmp_path / "wrong-continuable").resolve(),
        )
    with pytest.raises(TypeError, match="exactly PrimarySegmentRuntimeClosure"):
        terminal.produce_completed_primary_terminal(
            bundle,  # type: ignore[arg-type]
            runtime_closure=completed_closure.to_dict(),  # type: ignore[arg-type]
            inspection=inspection,
            evidence_directory=(tmp_path / "mapping-terminal").resolve(),
        )
    with pytest.raises(TypeError, match="private validator"):
        replace(completed).artifact_ref()

    reopened_completed = terminal.revalidate_completed_primary_terminal(
        bundle,  # type: ignore[arg-type]
        runtime_closure=completed_closure,
        evidence_directory=completed.evidence_directory,
        expected_manifest_file_sha256=completed.manifest_file_sha256,
        expected_raw_sacct_sha256=inspection.raw_sacct_sha256,
    )
    assert reopened_completed.to_dict() == completed.to_dict()
    reopened_continuable = terminal.revalidate_continuable_primary_terminal(
        bundle,  # type: ignore[arg-type]
        runtime_closure=continuable_closure,
        evidence_directory=continuable.evidence_directory,
        expected_manifest_file_sha256=continuable.manifest_file_sha256,
        expected_raw_sacct_sha256=inspection.raw_sacct_sha256,
    )
    assert reopened_continuable.to_dict() == continuable.to_dict()

    raw_path = (tmp_path / "post-job-primary-sacct.psv").resolve()
    raw_path.write_bytes(inspection.raw_bytes)
    monkeypatch.setattr(
        terminal,
        "reopen_verified_gate_p_operational_bundle",
        lambda _path, *, expected_file_sha256: bundle,
    )

    def poisoned(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("post-job primary finalizer touched live training/CUDA state")

    monkeypatch.setattr(terminal, "_validate_primary_dependencies", poisoned)
    from smart_reward import phase2_r3_materialization as materialization_module
    from smart_reward import phase2_r3_profile as profile_module

    for module, names in (
        (materialization_module, ("validate_r3_materialization",)),
        (
            profile_module,
            (
                "_require_live_cuda",
                "run_formal_gate_p_cuda_profile",
                "validate_formal_cuda_profile_result",
                "validate_gate_p_resource_plan",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, poisoned)
    finalized_completed_attempt = (tmp_path / "finalized-completed-attempt").resolve()
    (finalized_completed_attempt / "runtime-closures").mkdir(parents=True)
    (finalized_completed_attempt / "terminal-evidence").mkdir()
    finalized_completed_closure = finalized_completed_attempt / "runtime-closures" / "task-0.json"
    finalized_completed_closure.write_bytes(completed_closure.artifact_path.read_bytes())
    finalized = terminal.finalize_completed_primary_terminal_from_files(
        operational_bundle_path=(tmp_path / "bundle.json").resolve(),
        expected_operational_bundle_file_sha256=bundle.file_sha256,
        runtime_closure_path=finalized_completed_closure,
        expected_runtime_closure_file_sha256=completed_closure.file_sha256,
        raw_sacct_path=raw_path,
        expected_raw_sacct_sha256=inspection.raw_sacct_sha256,
        evidence_directory=(finalized_completed_attempt / "terminal-evidence" / "task-0-segment-1"),
    )
    assert finalized.artifact_ref().role == terminal.COMPLETED_PRIMARY_TERMINAL_ROLE
    finalized_continuable_attempt = (tmp_path / "finalized-continuable-attempt").resolve()
    (finalized_continuable_attempt / "runtime-closures").mkdir(parents=True)
    (finalized_continuable_attempt / "terminal-evidence").mkdir()
    finalized_continuable_closure = (
        finalized_continuable_attempt / "runtime-closures" / "task-0.json"
    )
    finalized_continuable_closure.write_bytes(continuable_closure.artifact_path.read_bytes())
    continued = terminal.finalize_continuable_primary_terminal_from_files(
        operational_bundle_path=(tmp_path / "bundle.json").resolve(),
        expected_operational_bundle_file_sha256=bundle.file_sha256,
        runtime_closure_path=finalized_continuable_closure,
        expected_runtime_closure_file_sha256=continuable_closure.file_sha256,
        raw_sacct_path=raw_path,
        expected_raw_sacct_sha256=inspection.raw_sacct_sha256,
        evidence_directory=(
            finalized_continuable_attempt / "terminal-evidence" / "task-0-segment-1"
        ),
    )
    assert continued.artifact_ref().role == CONTINUABLE_PRIMARY_TERMINAL_ROLE


def test_primary_closure_and_terminal_evidence_paths_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, closure = _publish_fake_primary_closure(
        tmp_path,
        monkeypatch,
        status="compute_complete_pending_external_scheduler_terminal",
        filename="canonical-attempt",
    )
    with pytest.raises(ValueError, match="terminal evidence must be"):
        terminal.produce_completed_primary_terminal(
            bundle,  # type: ignore[arg-type]
            runtime_closure=closure,
            inspection=_inspection(),
            evidence_directory=(
                closure.artifact_path.parent.parent / "terminal-evidence" / "wrong-segment-name"
            ),
        )

    admission, runtime, outcome, _ = _primary_fake_payloads(
        status="compute_complete_pending_external_scheduler_terminal"
    )
    with pytest.raises(ValueError, match="runtime-closures"):
        terminal.publish_primary_segment_runtime_closure(
            (tmp_path / "wrong-closure.json").resolve(),
            admission=admission,  # type: ignore[arg-type]
            runtime=runtime,  # type: ignore[arg-type]
            outcome=outcome,  # type: ignore[arg-type]
            operational_bundle=bundle,  # type: ignore[arg-type]
        )


def test_post_job_profile_finalizer_is_pure_data_and_uses_predeclared_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile resources may differ causally from the derived primary plan."""

    from smart_reward import oracle as oracle_module
    from smart_reward import phase2_r3_materialization as materialization_module
    from smart_reward import phase2_r3_profile as profile_module
    from smart_reward.phase2_r3_profile_artifacts import (
        publish_verified_gate_p_operational_bundle,
    )

    support_path = Path(__file__).with_name("test_phase2_r3_profile.py")
    spec = importlib.util.spec_from_file_location(
        "_phase2_r3_profile_test_support",
        support_path,
    )
    assert spec is not None and spec.loader is not None
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    conftest_path = Path(__file__).with_name("conftest.py")
    conftest_spec = importlib.util.spec_from_file_location(
        "_phase2_r3_conftest_support",
        conftest_path,
    )
    assert conftest_spec is not None and conftest_spec.loader is not None
    conftest_support = importlib.util.module_from_spec(conftest_spec)
    conftest_spec.loader.exec_module(conftest_support)

    intent = terminal.publish_profile_allocation_intent(
        (tmp_path / "profile-allocation-intent.json").resolve(),
        cluster="hpc4",
        account="sigroup",
        partition="gpu-l20",
        gpu_name="NVIDIA L20",
        gpus_per_task=1,
        cpus_per_task=2,
        memory_bytes=1024**3,
        requested_walltime_seconds=600,
    )
    science = support.science.__wrapped__()
    profile_run = support.profile_run.__wrapped__(
        science,
        conftest_support.sealed_r3_gate0_capability.__wrapped__(),
        conftest_support.sealed_r3_gate1_capabilities.__wrapped__(),
        conftest_support.seal_r3_train_materialization.__wrapped__(),
    )
    safety = support.safety_policy.__wrapped__(profile_run)
    envelope = support.envelope.__wrapped__(profile_run)
    preparation = support.preparation.__wrapped__(profile_run)
    core = support._core_profile(profile_run)
    support._patch_formal_runtime(monkeypatch, core)
    formal_result = profile_module.run_formal_gate_p_cuda_profile(
        profile_run,
        safety_policy=safety,
        envelope=envelope,
        preparation=preparation,
        io_probe_directory=tmp_path,
    )
    primary_plan = profile_module.build_gate_p_resource_plan(
        formal_result,
        safety_policy=safety,
        envelope=envelope,
        requested_walltime_seconds_per_segment=300,
        array_concurrency=2,
        cpus_per_task=4,
        memory_bytes=2 * 1024**3,
    )
    bundle = publish_verified_gate_p_operational_bundle(
        (tmp_path / "gate-p-operational-bundle.json").resolve(),
        profile_run=profile_run,
        safety_policy=safety,
        envelope=envelope,
        formal_result=formal_result,
        resource_plan=primary_plan,
    )
    assert intent.cpus_per_task != bundle.cpus_per_task
    assert intent.memory_bytes != bundle.memory_bytes
    assert intent.requested_walltime_seconds != bundle.requested_walltime_seconds_per_segment

    slurm = {
        "SLURM_CLUSTER_NAME": "hpc4",
        "SLURM_JOB_ID": "410001",
        "SLURM_ARRAY_JOB_ID": "410000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ACCOUNT": "sigroup",
        "SLURM_JOB_PARTITION": "gpu-l20",
        "SLURM_CPUS_PER_TASK": "2",
        "SLURM_GPUS_PER_TASK": "1",
        "SLURM_MEM_PER_NODE": "1024",
    }
    for name, value in slurm.items():
        monkeypatch.setenv(name, value)
    receipt = terminal.capture_profile_slurm_runtime_receipt(
        bundle,
        intent,
        (tmp_path / "profile-runtime-receipt.json").resolve(),
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    with pytest.raises(ValueError, match="SLURM_CPUS_PER_TASK"):
        terminal.capture_profile_slurm_runtime_receipt(
            bundle,
            intent,
            (tmp_path / "drifted-runtime-receipt.json").resolve(),
        )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(TypeError, match="publish/reopen"):
        replace(intent).validate_integrity()
    with pytest.raises(TypeError, match="capture/reopen"):
        replace(receipt).validate_integrity()

    raw = _raw_row(
        n_cpus="2",
        req_tres="billing=2,cpu=2,gres/gpu=1,mem=1G,node=1",
        alloc_tres=("billing=2,cpu=2,gres/gpu=1,mem=1G,node=1,gres/gpu:l20=1"),
    )
    raw_path = (tmp_path / "post-job-sacct.psv").resolve()
    raw_path.write_bytes(raw)

    def poisoned(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("post-job finalizer touched forbidden live state")

    for module, names in (
        (
            profile_module,
            (
                "_require_live_cuda",
                "run_formal_gate_p_cuda_profile",
                "validate_formal_cuda_profile_result",
                "validate_gate_p_resource_plan",
            ),
        ),
        (materialization_module, ("validate_r3_materialization",)),
        (
            oracle_module,
            (
                "fit_robust_oracle_transform",
                "pair_margins",
                "btl_probabilities",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, poisoned)

    capability = terminal.finalize_successful_profile_terminal_from_files(
        operational_bundle_path=bundle.artifact_path,
        expected_operational_bundle_file_sha256=bundle.file_sha256,
        allocation_intent_path=intent.artifact_path,
        expected_allocation_intent_file_sha256=intent.file_sha256,
        runtime_receipt_path=receipt.artifact_path,
        expected_runtime_receipt_file_sha256=receipt.file_sha256,
        raw_sacct_path=raw_path,
        expected_raw_sacct_sha256=_sha(raw),
        evidence_directory=(tmp_path / "profile-terminal").resolve(),
    )
    assert capability.artifact_ref().role == SUCCESSFUL_PROFILE_TERMINAL_ROLE
    assert capability.to_dict()["expected_slurm_resources"]["cpus_per_task"] == 2
    assert capability.to_dict()["resource_plan_sha256"] == (bundle.resource_plan_sha256)

    drifted_raw = _raw_row()
    drifted_path = (tmp_path / "drifted-sacct.psv").resolve()
    drifted_path.write_bytes(drifted_raw)
    with pytest.raises(ValueError, match="allocation identity|TRES"):
        terminal.finalize_successful_profile_terminal_from_files(
            operational_bundle_path=bundle.artifact_path,
            expected_operational_bundle_file_sha256=bundle.file_sha256,
            allocation_intent_path=intent.artifact_path,
            expected_allocation_intent_file_sha256=intent.file_sha256,
            runtime_receipt_path=receipt.artifact_path,
            expected_runtime_receipt_file_sha256=receipt.file_sha256,
            raw_sacct_path=drifted_path,
            expected_raw_sacct_sha256=_sha(drifted_raw),
            evidence_directory=(tmp_path / "drifted-terminal").resolve(),
        )


def test_locked_command_is_allocation_only_and_exact_task() -> None:
    command = terminal.sacct_terminal_command("410000_0")

    assert command[:6] == ("sacct", "-X", "-n", "-P", "-j", "410000_0")
    assert "ElapsedRaw" in command[-1]
    for invalid in ("410000_[0-2]", "410000.batch", "410000,410001", "410000%2"):
        with pytest.raises(ValueError, match="one exact"):
            terminal.sacct_terminal_command(invalid)
