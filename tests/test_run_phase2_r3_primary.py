from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hpc4" / "run_phase2_r3_primary.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_run_phase2_r3_primary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv() -> list[str]:
    return [
        "--project-root",
        "project",
        "--science-config",
        "science.yaml",
        "--source-config",
        "source.yaml",
        "--parent-registry",
        "registry.json",
        "--parent-registry-file-sha256",
        "1" * 64,
        "--gate0-file-sha256",
        "2" * 64,
        "--gate1-file-sha256",
        "3" * 64,
        "--source-test-receipt-file-sha256",
        "4" * 64,
        "--operational-bundle",
        "bundle.json",
        "--operational-bundle-file-sha256",
        "5" * 64,
        "--profile-allocation-intent",
        "profile-intent.json",
        "--profile-allocation-intent-file-sha256",
        "6" * 64,
        "--profile-runtime-receipt",
        "profile-runtime.json",
        "--profile-runtime-receipt-file-sha256",
        "7" * 64,
        "--profile-terminal-evidence-directory",
        "profile-terminal",
        "--profile-terminal-manifest-file-sha256",
        "8" * 64,
        "--profile-terminal-raw-sacct-sha256",
        "9" * 64,
        "--primary-submission-plan",
        "primary-plan.json",
        "--primary-submission-plan-file-sha256",
        "a" * 64,
        "--primary-submission-plan-sha256",
        "b" * 64,
        "--task-root",
        "task-root",
        "--runtime-closure",
        "runtime-closure.json",
    ]


@pytest.mark.parametrize(
    ("task_text", "expected"),
    [("0", (0, 20260801)), ("1", (1, 20260802)), ("2", (2, 20260803))],
)
def test_task_seed_map_is_derived_only_from_exact_slurm_array_task(
    monkeypatch: pytest.MonkeyPatch,
    task_text: str,
    expected: tuple[int, int],
) -> None:
    runner = _load_runner()
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", task_text)

    assert runner._task_seed_from_environment() == expected

    for invalid in ("00", "3", "20260801"):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", invalid)
        with pytest.raises(RuntimeError, match="SLURM_ARRAY_TASK_ID"):
            runner._task_seed_from_environment()


def test_parser_exposes_no_science_head_seed_or_continuation_override() -> None:
    runner = _load_runner()
    destinations = {action.dest for action in runner._parser()._actions}

    assert destinations.isdisjoint(
        {
            "task_id",
            "seed",
            "segment_index",
            "primary_heads",
            "head_order",
            "heldout",
            "control",
            "continuation_checkpoint",
            "continuation_terminal",
        }
    )
    with pytest.raises(SystemExit):
        runner._parser().parse_args([*_argv(), "--seed", "20260801"])


def test_segment1_runner_uses_exact_sealed_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")

    science = object()
    gate0 = object()
    gate1 = object()
    bundle = SimpleNamespace(requested_walltime_seconds_per_segment=300)
    profile_intent = object()
    profile_runtime = object()
    profile_terminal = object()
    authorization = object()
    design = SimpleNamespace(design_sha256="a" * 64)
    materialization_capability = object()
    materialized = SimpleNamespace(capability=materialization_capability)
    admission = SimpleNamespace(segment_index=1, admission_sha256="b" * 64)
    runtime = SimpleNamespace(runtime_sha256="c" * 64)
    outcome = SimpleNamespace(
        status="continuation_required_after_safe_checkpoint",
        outcome_sha256="d" * 64,
        file_sha256="e" * 64,
    )
    closure = SimpleNamespace(closure_sha256="f" * 64, file_sha256="0" * 64)
    identity_receipt = SimpleNamespace(file_sha256="1" * 64)
    segment_receipt = SimpleNamespace(
        file_sha256="2" * 64,
        payload={"segment_evidence_receipt_sha256": "3" * 64},
    )
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def bind(name: str, result: object):
        def call(*args: object, **kwargs: object) -> object:
            calls.append((name, args, kwargs))
            return result

        return call

    replacements = {
        "load_r3_science_config": bind("load_science", science),
        "verify_live_r3_gate0_in_container": bind("gate0", gate0),
        "verify_live_r3_gate1_in_container": bind("gate1", gate1),
        "reopen_verified_gate_p_operational_bundle": bind("bundle", bundle),
        "reopen_profile_allocation_intent": bind("profile_intent", profile_intent),
        "reopen_profile_slurm_runtime_receipt": bind(
            "profile_runtime",
            profile_runtime,
        ),
        "revalidate_successful_profile_terminal": bind(
            "profile_terminal",
            profile_terminal,
        ),
        "authorize_gate_p": bind("authorize", authorization),
        "create_r3_primary_design": bind("design", design),
        "materialize_r3_train_only_from_parent": bind(
            "materialize",
            materialized,
        ),
        "admit_primary_segment": bind("admit", admission),
        "publish_primary_identity_receipt": bind("identity_receipt", identity_receipt),
        "capture_slurm_segment_runtime": bind("runtime", runtime),
        "run_r3_primary_task_segment": bind("run_segment", outcome),
        "publish_primary_segment_runtime_closure": bind("closure", closure),
        "publish_segment_evidence_receipt": bind(
            "segment_evidence",
            segment_receipt,
        ),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(runner, name, replacement)

    class Signal:
        def __init__(self) -> None:
            calls.append(("signal_init", (), {}))

        def __enter__(self) -> Signal:
            calls.append(("signal_enter", (), {}))
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _traceback: object,
        ) -> None:
            calls.append(("signal_exit", (), {}))

    emitted: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "CheckpointSignal", Signal)
    monkeypatch.setattr(runner, "_emit", emitted.append)

    assert runner.main(_argv()) == 0
    assert [name for name, _, _ in calls] == [
        "load_science",
        "gate0",
        "gate1",
        "bundle",
        "profile_intent",
        "profile_runtime",
        "profile_terminal",
        "authorize",
        "design",
        "materialize",
        "admit",
        "identity_receipt",
        "runtime",
        "signal_init",
        "signal_enter",
        "run_segment",
        "signal_exit",
        "closure",
        "segment_evidence",
    ]
    by_name = {name: (args, kwargs) for name, args, kwargs in calls}
    assert by_name["gate0"][1] == {"expected_file_sha256": "2" * 64}
    assert by_name["gate1"][1] == {
        "expected_file_sha256": "3" * 64,
        "expected_source_test_receipt_file_sha256": "4" * 64,
    }
    assert by_name["bundle"] == (
        (Path("bundle.json"),),
        {"expected_file_sha256": "5" * 64},
    )
    assert by_name["profile_intent"] == (
        (Path("profile-intent.json"),),
        {"expected_file_sha256": "6" * 64},
    )
    assert by_name["profile_runtime"] == (
        (Path("profile-runtime.json"),),
        {
            "expected_file_sha256": "7" * 64,
            "operational_bundle": bundle,
            "allocation_intent": profile_intent,
        },
    )
    assert by_name["profile_terminal"] == (
        (bundle,),
        {
            "runtime_receipt": profile_runtime,
            "evidence_directory": Path("profile-terminal"),
            "expected_manifest_file_sha256": "8" * 64,
            "expected_raw_sacct_sha256": "9" * 64,
        },
    )
    assert by_name["authorize"][1] == {
        "operational_bundle": bundle,
        "successful_terminal": profile_terminal,
    }
    assert by_name["design"][1] == {
        "science": science,
        "gate0_capability": gate0,
        "gate1_capabilities": gate1,
        "profile_authorization": authorization,
        "operational_bundle": bundle,
    }
    assert by_name["materialize"][1]["seed"] == 20260802
    assert by_name["materialize"][1]["device"] == "cuda"
    assert by_name["materialize"][1]["expected_parent_registry_file_sha256"] == "1" * 64
    assert by_name["admit"][1] == {
        "design": design,
        "materialization_capability": materialization_capability,
        "task_id": 1,
        "seed": 20260802,
        "segment_index": 1,
        "continuation_evidence": None,
    }
    assert by_name["runtime"] == (
        (admission,),
        {"requested_walltime_seconds": 300},
    )
    assert by_name["run_segment"][0] == (admission,)
    assert by_name["run_segment"][1]["runtime"] is runtime
    assert by_name["run_segment"][1]["task_root"] == Path("task-root")
    assert isinstance(by_name["run_segment"][1]["checkpoint_signal"], Signal)
    assert by_name["run_segment"][1]["operational_policy"] is bundle
    assert by_name["closure"] == (
        (Path("runtime-closure.json"),),
        {
            "admission": admission,
            "runtime": runtime,
            "outcome": outcome,
            "operational_bundle": bundle,
        },
    )
    assert emitted == [
        {
            "status": "r3_primary_segment_closed_pending_external_scheduler_terminal",
            "segment_outcome_status": outcome.status,
            "task_id": 1,
            "seed": 20260802,
            "segment_index": 1,
            "design_sha256": design.design_sha256,
            "admission_sha256": admission.admission_sha256,
            "runtime_sha256": runtime.runtime_sha256,
            "segment_outcome_sha256": outcome.outcome_sha256,
            "segment_outcome_file_sha256": outcome.file_sha256,
            "runtime_closure_sha256": closure.closure_sha256,
            "runtime_closure_file_sha256": closure.file_sha256,
            "primary_identity_receipt_file_sha256": identity_receipt.file_sha256,
            "segment_evidence_receipt_file_sha256": segment_receipt.file_sha256,
            "segment_evidence_receipt_sha256": (
                segment_receipt.payload["segment_evidence_receipt_sha256"]
            ),
            "external_scheduler_terminal_required": True,
        }
    ]
