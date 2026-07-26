from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hpc4" / "phase2_r3_terminalize_stdlib.py"
SPEC = importlib.util.spec_from_file_location("_phase2_r3_terminalize_stdlib", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
terminalize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(terminalize)


def _canonical(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _embedded(value: dict[str, object]) -> dict[str, object]:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    return {
        "encoding": "canonical-json-utf8-newline",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "payload": value,
    }


def _gatep_attempt(tmp_path: Path) -> tuple[Path, Path]:
    project = (tmp_path / "project").resolve()
    attempt = project / "runs" / "phase2-recovery-r3" / "gatep" / ("a" * 40) / "gatep-attempt-001"
    attempt.mkdir(parents=True)
    _canonical(attempt / "gatep-operational-bundle.json", {})
    _canonical(attempt / "profile-allocation-intent.json", {})
    _canonical(
        attempt / "profile-runtime-receipt.json",
        {
            "slurm_runtime": {
                "job_id": "12345",
                "array_job_id": None,
                "array_task_id": None,
            }
        },
    )
    return project, attempt


def test_gatep_route_is_fixed_to_attempt_namespace(tmp_path: Path) -> None:
    project, attempt = _gatep_attempt(tmp_path)
    route = terminalize.plan_gatep(project_root=project, attempt_root=attempt)
    assert route == {
        "mode": "gatep",
        "job_selector": "12345",
        "route_status": "profile",
        "finalizer_command": "profile-finalize",
        "attempt_root": os.fspath(attempt),
        "operational_bundle": os.fspath(attempt / "gatep-operational-bundle.json"),
        "allocation_intent": os.fspath(attempt / "profile-allocation-intent.json"),
        "runtime_receipt": os.fspath(attempt / "profile-runtime-receipt.json"),
        "runtime_closure": None,
        "raw_sacct": os.fspath(attempt / "terminal-raw" / "profile.sacct.psv"),
        "evidence_directory": os.fspath(attempt / "terminal-evidence" / "profile"),
        "task_id": None,
        "segment_index": None,
    }


def test_gatep_route_rejects_noncanonical_attempt_number(tmp_path: Path) -> None:
    project, attempt = _gatep_attempt(tmp_path)
    invalid = attempt.with_name("gatep-attempt-01")
    attempt.rename(invalid)
    with pytest.raises(ValueError, match="gatep-attempt-NNN"):
        terminalize.plan_gatep(project_root=project, attempt_root=invalid)


@pytest.mark.parametrize(
    ("status", "command"),
    [
        (
            "continuation_required_after_safe_checkpoint",
            "primary-continuable-finalize",
        ),
        (
            "compute_complete_pending_external_scheduler_terminal",
            "primary-completed-finalize",
        ),
    ],
)
def test_primary_route_uses_closure_status_and_segment(
    tmp_path: Path,
    status: str,
    command: str,
) -> None:
    project, gatep = _gatep_attempt(tmp_path)
    attempt = (
        project / "runs" / "phase2-recovery-r3" / "primary" / ("b" * 40) / "primary-attempt-001"
    )
    (attempt / "runtime-closures").mkdir(parents=True)
    admission = {"task_id": 2, "segment_index": 3}
    runtime = {
        "task_id": 2,
        "segment_index": 3,
        "job_id": "45678",
        "array_job_id": "45670",
        "array_task_id": 2,
    }
    _canonical(
        attempt / "runtime-closures" / "task-2.json",
        {
            "admission": _embedded(admission),
            "runtime": _embedded(runtime),
            "status": status,
        },
    )
    route = terminalize.plan_primary(
        project_root=project,
        attempt_root=attempt,
        task_id=2,
        operational_bundle=gatep / "gatep-operational-bundle.json",
    )
    assert route["job_selector"] == "45670_2"
    assert route["job_id_raw"] == "45678"
    assert route["finalizer_command"] == command
    assert route["segment_index"] == 3
    assert route["raw_sacct"] == os.fspath(attempt / "terminal-raw" / "task-2-segment-3.sacct.psv")
    assert route["evidence_directory"] == os.fspath(
        attempt / "terminal-evidence" / "task-2-segment-3"
    )


def test_raw_capture_is_no_overwrite_and_exactly_reentrant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (tmp_path / "raw" / "task.sacct.psv").resolve()
    output.parent.mkdir()
    row = b"123_0|456|COMPLETED|0:0|0:0|hpc4|sigroup|gpu-l20|l20_qos|1|8|req|alloc|42\n"

    def run(command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(command, 0, row, b"")

    monkeypatch.setattr(terminalize, "_run_command", run)
    first = terminalize.capture_raw_sacct(
        job_selector="123_0",
        output=output,
        user="yyangjo",
        attempts=1,
        interval_seconds=0,
    )
    assert first["reused"] is False
    assert output.read_bytes() == row
    second = terminalize.capture_raw_sacct(
        job_selector="123_0",
        output=output,
        user="yyangjo",
        attempts=1,
        interval_seconds=0,
    )
    assert second["reused"] is True
    assert second["raw_sha256"] == first["raw_sha256"]

    changed = row.replace(b"COMPLETED", b"FAILED   ")

    def drifted(command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(command, 0, changed, b"")

    monkeypatch.setattr(terminalize, "_run_command", drifted)
    with pytest.raises(ValueError, match="differ"):
        terminalize.capture_raw_sacct(
            job_selector="123_0",
            output=output,
            user="yyangjo",
            attempts=1,
            interval_seconds=0,
        )
    assert output.read_bytes() == row


def test_raw_capture_rejects_a_live_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        assert command[0] == "squeue"
        return subprocess.CompletedProcess(command, 0, b"123_0\n999\n", b"")

    monkeypatch.setattr(terminalize, "_run_command", run)
    with pytest.raises(RuntimeError, match="still present"):
        terminalize.capture_raw_sacct(
            job_selector="123_0",
            output=(tmp_path / "raw.psv").resolve(),
            user="yyangjo",
            attempts=1,
            interval_seconds=0,
        )


def test_locked_sacct_query_matches_r3_terminal_contract() -> None:
    assert terminalize.sacct_terminal_command("123_2") == (
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        "123_2",
        "--format="
        "JobID%64,JobIDRaw%32,State%64,ExitCode%32,DerivedExitCode%32,"
        "Cluster%64,Account%64,Partition%64,QOS%64,NNodes%16,NCPUS%16,"
        "ReqTRES%512,AllocTRES%512,ElapsedRaw%32",
    )
