from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

from smart_reward import phase2_r3_gatep_failure as failure

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "hpc4" / "capture_phase2_r3_gatep_failure.py"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source(commit: str) -> dict[str, str]:
    return {
        "source_git_commit": commit,
        "gate0_file_sha256": _digest(f"{commit}:gate0"),
        "gate1_file_sha256": _digest(f"{commit}:gate1"),
        "source_test_receipt_file_sha256": _digest(f"{commit}:source-test"),
        "science_config_file_sha256": _digest("science"),
        "container_file_sha256": _digest("container"),
    }


def _raw_sacct(job_id: str, *, state: str = "FAILED") -> bytes:
    return (
        "|".join(
            (
                job_id,
                job_id,
                state,
                "1:0",
                "0:0",
                "hpc4",
                "sigroup",
                "gpu-l20",
                "l20_qos",
                "1",
                "8",
                "billing=8,cpu=8,gres/gpu=1,mem=64G,node=1",
                "billing=8,cpu=8,gres/gpu=1,gres/gpu:l20=1,mem=64G,node=1",
                "8",
            )
        )
        + "\n"
    ).encode()


def _attempt(
    project: Path,
    *,
    identity: str,
    index: int,
    job_id: str,
) -> tuple[Path, Path, Path]:
    attempt = (
        project / "runs" / "phase2-recovery-r3" / "gatep" / identity / f"gatep-attempt-{index:03d}"
    )
    logs = attempt / "logs"
    logs.mkdir(parents=True)
    stdout = logs / f"gatep-{job_id}.out"
    stderr = logs / f"gatep-{job_id}.err"
    stdout.write_bytes(b"")
    stderr.write_bytes(b"Gate-0 committed capture source changed after publication\n")
    (attempt / "profile-allocation-intent.json").write_bytes(b'{"legacy":true}\n')
    return attempt.resolve(), stdout.resolve(), stderr.resolve()


def _publish_failure(
    tmp_path: Path,
    *,
    commit: str = "7" * 40,
    identity: str = "legacy-7491b17",
    index: int = 1,
    job_id: str = "1657236",
) -> tuple[Path, failure.GatePFailureReceipt]:
    project = (tmp_path / "project").resolve()
    project.mkdir(exist_ok=True)
    attempt, stdout, stderr = _attempt(
        project,
        identity=identity,
        index=index,
        job_id=job_id,
    )
    raw = (tmp_path / f"{job_id}.sacct.psv").resolve()
    raw.write_bytes(_raw_sacct(job_id))
    receipt = failure.publish_gate_p_failure_receipt(
        project_root=project,
        attempt_root=attempt,
        raw_sacct_path=raw,
        stdout_path=stdout,
        stderr_path=stderr,
        job_id=job_id,
        failure_stage="gate0_revalidation",
        captured_at_utc="2026-07-27T12:00:00Z",
        **_source(commit),
    )
    return project, receipt


def test_failure_receipt_is_append_only_non_authorizing_and_byte_closed(
    tmp_path: Path,
) -> None:
    project, receipt = _publish_failure(tmp_path)

    assert receipt.artifact_path.name == failure.FAILURE_RECEIPT_FILENAME
    assert receipt.artifact_path.parent.name == failure.FAILURE_EVIDENCE_DIRECTORY
    assert {path.name for path in receipt.artifact_path.parent.iterdir()} == {
        failure.FAILURE_RAW_SACCT_FILENAME,
        failure.FAILURE_RECEIPT_FILENAME,
    }
    payload = receipt.payload
    assert payload["source_binding_evidence"] == {
        "classification": ("operator_declared_historical_hashes_bound_as_identity_only"),
        "mechanically_reverified_source_files_by_failure_receipt": False,
    }
    assert payload["authority"] == {
        "authorizes_gate_p_success": False,
        "authorizes_retry": False,
        "authorizes_primary": False,
        "authorizes_science_change": False,
        "reusable_training_state": False,
    }
    assert payload["scheduler"]["parsed_sacct"]["formal_claim_eligible"] is False
    assert payload["scheduler"]["parsed_sacct"]["row"]["state"] == "FAILED"
    assert payload["failure"]["stage"] == "gate0_revalidation"
    assert payload["logs"]["stdout"]["sha256"] == hashlib.sha256(b"").hexdigest()
    assert payload["failure"]["publication_state"] == {
        "profile_allocation_intent_present": True,
        "operational_bundle_present": False,
        "runtime_receipt_present": False,
        "successful_terminal_evidence_present": False,
    }
    reopened = failure.reopen_gate_p_failure_receipt(
        receipt.artifact_path,
        project_root=project,
        expected_file_sha256=receipt.file_sha256,
    )
    assert reopened.receipt_sha256 == receipt.receipt_sha256

    with pytest.raises(FileExistsError, match="overwrite"):
        failure.publish_gate_p_failure_receipt(
            project_root=project,
            attempt_root=receipt.artifact_path.parent.parent,
            raw_sacct_path=receipt.artifact_path.parent / failure.FAILURE_RAW_SACCT_FILENAME,
            stdout_path=project
            / payload["attempt"]["project_relative"]
            / payload["logs"]["stdout"]["relative"],
            stderr_path=project
            / payload["attempt"]["project_relative"]
            / payload["logs"]["stderr"]["relative"],
            job_id=receipt.job_id,
            failure_stage="gate0_revalidation",
            captured_at_utc="2026-07-27T12:00:00Z",
            **_source("7" * 40),
        )


def test_failure_receipt_reopen_rejects_log_or_inventory_mutation(tmp_path: Path) -> None:
    project, receipt = _publish_failure(tmp_path)
    stderr = receipt.artifact_path.parent.parent / "logs" / f"gatep-{receipt.job_id}.err"
    stderr.write_bytes(b"mutated\n")

    with pytest.raises(ValueError, match="attempt changed"):
        failure.reopen_gate_p_failure_receipt(
            receipt.artifact_path,
            project_root=project,
            expected_file_sha256=receipt.file_sha256,
        )


def test_cross_campaign_lineage_resets_to_001_and_binds_both_receipt_hashes(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    tmp_path = tmp_path_factory.mktemp("g")
    project, predecessor = _publish_failure(tmp_path)
    current = _source("8" * 40)
    plan = failure.plan_next_gate_p_attempt(
        project_root=project,
        predecessor_receipt=predecessor,
        **current,
    )
    assert plan["attempt_index"] == 1
    assert plan["predecessor_relation"] == (
        "new_campaign_attempt_001_with_cross_campaign_predecessor"
    )
    identity = str(plan["campaign_identity_sha256"])
    attempt = project / "runs" / "phase2-recovery-r3" / "gatep" / identity / "gatep-attempt-001"
    (attempt / "logs").mkdir(parents=True)

    lineage = failure.publish_gate_p_attempt_lineage(
        project_root=project,
        attempt_root=attempt.resolve(),
        predecessor_receipt=predecessor,
        **current,
    )

    assert lineage.artifact_path == attempt.resolve() / failure.ATTEMPT_LINEAGE_FILENAME
    assert lineage.campaign_identity_sha256 == identity
    assert lineage.attempt_index == 1
    assert lineage.predecessor_file_sha256 == predecessor.file_sha256
    assert lineage.predecessor_receipt_sha256 == predecessor.receipt_sha256
    assert lineage.payload["authority"]["authorizes_retry"] is False
    assert lineage.profile_intent_binding() == {
        "attempt_lineage_file_sha256": lineage.file_sha256,
        "attempt_lineage_sha256": lineage.lineage_sha256,
    }
    reopened = failure.reopen_gate_p_attempt_lineage(
        lineage.artifact_path,
        project_root=project,
        expected_file_sha256=lineage.file_sha256,
    )
    assert reopened.lineage_sha256 == lineage.lineage_sha256


def test_same_campaign_requires_exact_next_index(tmp_path: Path) -> None:
    project, predecessor = _publish_failure(tmp_path, commit="8" * 40)
    current = _source("8" * 40)
    plan = failure.plan_next_gate_p_attempt(
        project_root=project,
        predecessor_receipt=predecessor,
        **current,
    )
    assert plan["attempt_index"] == 2
    assert plan["attempt_name"] == "gatep-attempt-002"
    identity = str(plan["campaign_identity_sha256"])
    wrong = project / "runs" / "phase2-recovery-r3" / "gatep" / identity / "gatep-attempt-003"
    (wrong / "logs").mkdir(parents=True)
    with pytest.raises(ValueError, match="campaign/predecessor"):
        failure.publish_gate_p_attempt_lineage(
            project_root=project,
            attempt_root=wrong.resolve(),
            predecessor_receipt=predecessor,
            **current,
        )


def test_receipt_rejects_completed_zero_exit_or_wrong_job(tmp_path: Path) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    attempt, stdout, stderr = _attempt(
        project,
        identity="legacy",
        index=1,
        job_id="1657236",
    )
    raw = (tmp_path / "raw.psv").resolve()
    raw.write_bytes(_raw_sacct("1657236", state="COMPLETED"))
    with pytest.raises(ValueError, match="terminal failure"):
        failure.publish_gate_p_failure_receipt(
            project_root=project,
            attempt_root=attempt,
            raw_sacct_path=raw,
            stdout_path=stdout,
            stderr_path=stderr,
            job_id="1657236",
            failure_stage="gate0_revalidation",
            **_source("7" * 40),
        )

    raw.write_bytes(_raw_sacct("1657237"))
    with pytest.raises(ValueError, match="does not identify"):
        failure.publish_gate_p_failure_receipt(
            project_root=project,
            attempt_root=attempt,
            raw_sacct_path=raw,
            stdout_path=stdout,
            stderr_path=stderr,
            job_id="1657236",
            failure_stage="gate0_revalidation",
            **_source("7" * 40),
        )


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_gatep_failure_cli", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_has_no_output_override_and_emits_campaign_identity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    arguments = [
        "derive-campaign",
        "--source-git-commit",
        "8" * 40,
        "--gate0-file-sha256",
        _digest("gate0"),
        "--gate1-file-sha256",
        _digest("gate1"),
        "--source-test-receipt-file-sha256",
        _digest("source-test"),
        "--science-config-file-sha256",
        _digest("science"),
        "--container-file-sha256",
        _digest("container"),
    ]
    assert cli.main(arguments) == 0
    output = capsys.readouterr().out
    assert '"status":"r3_gate_p_campaign_identity_derived_non_authorizing"' in output
    assert '"campaign_identity_sha256":' in output
    parser = cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*arguments, "--output", os.fspath(Path("mutable.json"))])


def test_gatep_submission_and_compute_fail_closed_over_predecessor_lineage() -> None:
    launcher = (ROOT / "scripts" / "hpc4" / "submit_phase2_r3_gatep.sh").read_text(encoding="utf-8")
    submitter = (ROOT / "scripts" / "hpc4" / "phase2_r3_gatep_submission.sbatch").read_text(
        encoding="utf-8"
    )
    compute = (ROOT / "scripts" / "hpc4" / "phase2_r3_gatep.sbatch").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "hpc4" / "run_phase2_r3_gatep.py").read_text(encoding="utf-8")
    for text in (launcher, submitter):
        assert "PRORM_R3_GATEP_PREDECESSOR_FAILURE_RECEIPT" in text
        assert "PRORM_R3_GATEP_PREDECESSOR_FAILURE_RECEIPT_FILE_SHA256" in text
    assert submitter.index("plan-next") < submitter.index('mkdir -- "${attempt_root}"')
    assert submitter.index("plan-next") < submitter.index(
        'mkdir -m 2750 -- "${attempt_parent_text}"'
    )
    assert 'attempt_parent_text="${gatep_root}/${campaign_identity_sha256}"' in submitter
    assert '[[ -d "${attempt_parent_text}" && ! -L "${attempt_parent_text}" ]]' in submitter
    assert '"$(stat -c \'%a\' -- "${attempt_parent}")" == "2750"' in submitter
    assert submitter.index("publish-lineage") < submitter.index(
        'python3 "${terminal_cli}" profile-intent'
    )
    assert submitter.index("--attempt-lineage-file-sha256") < submitter.index("sbatch \\")
    assert "PRORM_R3_GATEP_ATTEMPT_LINEAGE_FILE_SHA256" in submitter
    assert "PRORM_R3_GATEP_ATTEMPT_LINEAGE_SHA256" in submitter
    assert "PRORM_R3_GATEP_ATTEMPT_LINEAGE_FILE_SHA256" in compute
    assert "PRORM_R3_GATEP_ATTEMPT_LINEAGE_SHA256" in compute
    assert compute.index("inspect-lineage") < compute.index(
        'runner="${repo_root}/scripts/hpc4/run_phase2_r3_gatep.py"'
    )
    assert '--attempt-lineage "${attempt_lineage}"' in compute
    assert (
        '--attempt-lineage-file-sha256 \\\n      "${PRORM_R3_GATEP_ATTEMPT_LINEAGE_FILE_SHA256}"'
    ) in compute
    assert (
        '--attempt-lineage-sha256 \\\n      "${PRORM_R3_GATEP_ATTEMPT_LINEAGE_SHA256}"'
    ) in compute
    assert (
        "${project_root}/runs/phase2-recovery-r3/gate1/"
        "${PRORM_R3_GIT_COMMIT}/r3-source-test-receipt.json"
    ) in compute
    runner_main = runner.split("def main(", maxsplit=1)[1]
    assert runner_main.index("reopen_gate_p_attempt_lineage(") < runner_main.index(
        "run_formal_gate_p_cuda_profile("
    )
    assert runner_main.index("_validate_lineage_cross_binding(") < runner_main.index(
        "run_formal_gate_p_cuda_profile("
    )


def test_gatep_runner_rejects_intent_actual_or_exported_lineage_mismatch() -> None:
    spec = importlib.util.spec_from_file_location(
        "_gatep_runner",
        ROOT / "scripts" / "hpc4" / "run_phase2_r3_gatep.py",
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    runner._validate_lineage_cross_binding(
        intent_lineage_file_sha256="1" * 64,
        intent_lineage_sha256="2" * 64,
        actual_lineage_file_sha256="1" * 64,
        actual_lineage_sha256="2" * 64,
        exported_lineage_file_sha256="1" * 64,
        exported_lineage_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        runner._validate_lineage_cross_binding(
            intent_lineage_file_sha256="1" * 64,
            intent_lineage_sha256="2" * 64,
            actual_lineage_file_sha256="3" * 64,
            actual_lineage_sha256="2" * 64,
            exported_lineage_file_sha256="3" * 64,
            exported_lineage_sha256="2" * 64,
        )
    with pytest.raises(ValueError, match="semantic SHA-256 differs"):
        runner._validate_lineage_cross_binding(
            intent_lineage_file_sha256="1" * 64,
            intent_lineage_sha256="2" * 64,
            actual_lineage_file_sha256="1" * 64,
            actual_lineage_sha256="4" * 64,
            exported_lineage_file_sha256="1" * 64,
            exported_lineage_sha256="4" * 64,
        )
