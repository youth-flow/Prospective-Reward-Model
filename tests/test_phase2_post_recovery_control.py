from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

import smart_reward.phase2_post_recovery_control as control

ROOT = Path(__file__).resolve().parents[1]


def _sacct_raw(array_job_id: str = "2000") -> bytes:
    return "".join(
        f"{array_job_id}_{task}|{7000 + task}|COMPLETED|0:0|0:0|hpc4|sigroup|"
        "gpu-l20|1|8|billing=8,cpu=8,gres/gpu=1,mem=96G,node=1|"
        "billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1\n"
        for task in range(3)
    ).encode()


def _load_recovery_test_helpers():
    path = ROOT / "tests" / "test_phase2_recovery_aggregate.py"
    spec = importlib.util.spec_from_file_location("_recovery_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_capture_preserves_exact_raw_bytes_and_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _sacct_raw()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=raw,
            stderr=b"",
        ),
    )
    output = tmp_path / "terminal.json"

    payload = control.capture_post_recovery_terminal_evidence(
        "2000",
        output,
        pilot_phase="calibration",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    verified = control.verify_post_recovery_terminal_evidence(
        output,
        expected_sha256=digest,
        expected_array_job_id="2000",
        expected_pilot_phase="calibration",
    )

    assert payload == verified
    assert (tmp_path / "terminal.sacct.psv").read_bytes() == raw
    assert [row["job_id_raw"] for row in payload["rows"]] == [
        "7000",
        "7001",
        "7002",
    ]
    assert payload["query"][-1] == (
        "--format=JobID%32,JobIDRaw%32,State%64,ExitCode%32,"
        "DerivedExitCode%32,Cluster%64,Account%64,Partition%64,"
        "NNodes%16,NCPUS%16,ReqTRES%512,AllocTRES%512"
    )
    assert payload["rows"][0]["n_nodes"] == 1
    assert payload["rows"][0]["n_cpus"] == 8
    with pytest.raises(ValueError, match="identity"):
        control.verify_post_recovery_terminal_evidence(
            output,
            expected_sha256=digest,
            expected_array_job_id="2000",
            expected_pilot_phase="freeze",
        )


@pytest.mark.parametrize(
    "raw",
    [
        _sacct_raw().replace(b"COMPLETED", b"FAILED", 1),
        _sacct_raw().replace(b"0:0", b"1:0", 1),
        _sacct_raw().replace(b"hpc4", b"other", 1),
        _sacct_raw()
        + (
            b"2000.batch|9000|COMPLETED|0:0|0:0|hpc4|sigroup|gpu-l20|1|8|"
            b"billing=8,cpu=8,gres/gpu=1,mem=96G,node=1|"
            b"billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1\n"
        ),
        _sacct_raw().replace(b"2000_1", b"2000_[1-2]", 1),
        _sacct_raw().replace(b"7001", b"7000", 1),
        _sacct_raw().replace(b"|1|8|billing=", b"|2|8|billing=", 1),
        _sacct_raw().replace(b"|1|8|billing=", b"|1|16|billing=", 1),
        _sacct_raw().replace(b"gres/gpu=1", b"gres/gpu=2", 1),
        _sacct_raw().replace(b"gres/gpu:l20=1", b"gres/gpu:a100=1", 1),
        _sacct_raw().replace(b"node=1\n", b"node=1|\n", 1),
    ],
)
def test_terminal_capture_rejects_tampered_or_extra_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=raw,
            stderr=b"",
        ),
    )
    with pytest.raises(ValueError, match="exact|three|twelve"):
        control.capture_post_recovery_terminal_evidence(
            "2000",
            tmp_path / "bad.json",
            pilot_phase="calibration",
        )


def test_terminal_capture_rejects_noncanonical_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    noncanonical = real / ".." / "terminal.json"
    with pytest.raises(ValueError, match="parent must be"):
        control._write_exclusive(noncanonical, b"x", name="test evidence")


def test_terminal_parser_rejects_duplicate_allocation_job_ids() -> None:
    duplicate = _sacct_raw().replace(b"7001", b"7000", 1)
    with pytest.raises(ValueError, match="exact successful HPC4 allocation"):
        control._parse_sacct_raw(duplicate, array_job_id="2000")


def test_terminal_pair_rolls_back_raw_if_json_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _sacct_raw()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=raw,
            stderr=b"",
        ),
    )
    original = control._write_exclusive
    calls = 0

    def injected(path: Path, value: bytes, *, name: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected JSON publication failure")
        original(path, value, name=name)

    monkeypatch.setattr(control, "_write_exclusive", injected)
    with pytest.raises(OSError, match="injected"):
        control.capture_post_recovery_terminal_evidence(
            "2000",
            tmp_path / "terminal.json",
            pilot_phase="calibration",
        )
    assert not (tmp_path / "terminal.json").exists()
    assert not (tmp_path / "terminal.sacct.psv").exists()


def _success_marker(tmp_path: Path, **overrides: str) -> Path:
    fields = {
        "schema_version": control.POST_RECOVERY_RUN_STATUS_SCHEMA,
        "status": "SUCCESS",
        "pilot_phase": "calibration",
        "workload_exit_code": "0",
        "final_exit_code": "0",
        "slurm_job_id": "7000",
        "allocation_job_id_raw": "7000",
        "slurm_array_task_job_id": "2000_0",
        "array_job_id": "2000",
        "array_task_id": "0",
        "seed": "20260801",
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "gpu-l20",
        "restart_count": "0",
        "phase2_design_sha256": "a" * 64,
        "base_config_hash": "b" * 64,
        "git_commit": "c" * 40,
        "recovery_authorization_sha256": "d" * 64,
        "optimizer_schedule_sha256": control.OPTIMIZER_SCHEDULE_SHA256,
        "submission_intent_sha256": "3" * 64,
        "submission_ledger_sha256": "4" * 64,
        "materialization_mode": "fresh",
        "recovery_outputs_mounted": "false",
        "hf_root_mount_mode": "read_only",
        "datasets_cache_scope": "job_local",
        "artifact_metadata_sha256": "e" * 64,
        "phase2_result_sha256": "f" * 64,
        "phase2_output_verification_sha256": "1" * 64,
        "post_recovery_output_verification_sha256": "2" * 64,
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    fields.update(overrides)
    marker = tmp_path / "SUCCESS"
    marker.write_text(
        "".join(f"{key}={value}\n" for key, value in fields.items()),
        encoding="utf-8",
        newline="\n",
    )
    return marker


def test_success_marker_binds_fresh_head_free_scheduler_identity(
    tmp_path: Path,
) -> None:
    marker = _success_marker(tmp_path)
    value = control.verify_post_recovery_success_marker(
        marker,
        expected_array_job_id="2000",
        expected_task_id=0,
        expected_seed=20260801,
        expected_design_sha256="a" * 64,
        expected_base_config_hash="b" * 64,
        expected_git_commit="c" * 40,
        expected_authorization_sha256="d" * 64,
        expected_submission_intent_sha256="3" * 64,
        expected_submission_ledger_sha256="4" * 64,
        expected_allocation_job_id_raw="7000",
        expected_pilot_phase="calibration",
    )
    assert value["materialization_mode"] == "fresh"
    assert value["recovery_outputs_mounted"] == "false"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "FAILED"),
        ("pilot_phase", "freeze"),
        ("restart_count", "1"),
        ("slurm_job_id", "7001"),
        ("allocation_job_id_raw", "7001"),
        ("slurm_array_task_job_id", "2000_1"),
        ("materialization_mode", "reused"),
        ("recovery_outputs_mounted", "true"),
        ("optimizer_schedule_sha256", "0" * 64),
    ],
)
def test_success_marker_tampering_fails_closed(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    marker = _success_marker(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        control.verify_post_recovery_success_marker(
            marker,
            expected_array_job_id="2000",
            expected_task_id=0,
            expected_seed=20260801,
            expected_design_sha256="a" * 64,
            expected_base_config_hash="b" * 64,
            expected_git_commit="c" * 40,
            expected_authorization_sha256="d" * 64,
            expected_submission_intent_sha256="3" * 64,
            expected_submission_ledger_sha256="4" * 64,
            expected_allocation_job_id_raw="7000",
            expected_pilot_phase="calibration",
        )


def test_success_marker_rejects_terminal_allocation_job_id_mismatch(
    tmp_path: Path,
) -> None:
    marker = _success_marker(tmp_path)
    with pytest.raises(ValueError, match="slurm_job_id"):
        control.verify_post_recovery_success_marker(
            marker,
            expected_array_job_id="2000",
            expected_task_id=0,
            expected_seed=20260801,
            expected_design_sha256="a" * 64,
            expected_base_config_hash="b" * 64,
            expected_git_commit="c" * 40,
            expected_authorization_sha256="d" * 64,
            expected_submission_intent_sha256="3" * 64,
            expected_submission_ledger_sha256="4" * 64,
            expected_allocation_job_id_raw="7999",
            expected_pilot_phase="calibration",
        )


def test_real_recovery_authorization_verifier_rejects_byte_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers = _load_recovery_test_helpers()
    paths, scheduler, output, _ = helpers._campaign(tmp_path, monkeypatch)
    helpers.aggregate.write_phase2_recovery_authorization(
        paths,
        output,
        scheduler_evidence=scheduler,
        aggregator_git_commit=helpers.AGGREGATOR_COMMIT,
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    payload = control.verify_recovery_authorization_file(
        output,
        expected_sha256=digest,
    )
    assert payload["source_array_job_id"] == "1648125"

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["full_calibration_authorized"] = False
    output.write_bytes(helpers.aggregate._canonical_bytes(tampered))
    with pytest.raises(ValueError, match="SHA256"):
        control.verify_recovery_authorization_file(
            output,
            expected_sha256=digest,
        )
