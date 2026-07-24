from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
VALIDATOR = ROOT / "scripts" / "hpc4" / "validate_phase2_terminal.py"
COMPUTE_TERMINALIZER = ROOT / "scripts" / "hpc4" / "terminalize_phase2_compute_failure.sh"
SCHEDULER_TERMINALIZER = ROOT / "scripts" / "hpc4" / "terminalize_phase2_scheduler_failure.sh"
PUBLISHER = ROOT / "scripts" / "hpc4" / "publish_phase2_terminal_bundle.py"


def _write_json(path: Path, value: object) -> bytes:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path.write_bytes(raw)
    return raw


def _write_claim(
    path: Path,
    *,
    design: str,
    base: str,
    freeze: str,
    commit: str,
    submission_sha: str,
    scheduler_registry_sha: str,
) -> None:
    path.write_text(
        "\n".join(
            (
                "schema_version=prorm-phase2-formal-scheduler-attempt-claim/v1",
                "status=CLAIMED_BY_SCHEDULER_RECONCILIATION",
                "cluster_name=hpc4",
                "array_job_id=900",
                "array_task_id=0",
                "slurm_job_id=777",
                "slurm_restart_count=0",
                "attempt_index=1",
                "seed=20260901",
                f"phase2_design_sha256={design}",
                f"base_config_hash={base}",
                f"git_commit={commit}",
                f"accepted_freeze_aggregate_sha256={freeze}",
                f"registry_submission_sha256={submission_sha}",
                "registry_execution_sha256=none",
                f"registry_scheduler_terminal_sha256={scheduler_registry_sha}",
                "created_at_utc=2026-07-25T00:00:00Z",
                "",
            )
        ),
        encoding="utf-8",
    )


def _scheduler_terminal_tree(tmp_path: Path) -> tuple[Path, list[str]]:
    design = "a" * 64
    base = "b" * 64
    runtime = "c" * 64
    commit = "d" * 40
    image = "e" * 64
    inventory = "f" * 64
    freeze = "1" * 64
    job = tmp_path / design / "seed-20260901" / "attempt-1" / "job-900_0"
    job.mkdir(parents=True)
    scheduler_raw = b"hpc4|777|900_0|NODE_FAIL|1:0\n"
    (job / "scheduler-terminal-attestation.raw").write_bytes(scheduler_raw)
    scheduler_sha = hashlib.sha256(scheduler_raw).hexdigest()
    registry = job.parent.parent.parent / "campaign-registry"
    for name in ("submissions", "executions", "recoveries", "scheduler-terminals"):
        (registry / name).mkdir(parents=True, exist_ok=True)
    submission = {
        "schema_version": "prorm-phase2-campaign-submission/v1",
        "status": "committed_while_slurm_held",
        "phase2_design_sha256": design,
        "base_config_hash": base,
        "git_commit": commit,
        "accepted_freeze_aggregate_sha256": freeze,
        "array_job_id": "900",
        "submitted_cluster": "hpc4",
        "array_spec": "0",
        "attempt_index": 1,
        "entries": [
            {
                "seed": 20260901,
                "attempt_index": 1,
                "array_job_id": "900",
                "array_task_id": 0,
            }
        ],
        "job_tuple": {},
        "producer": {},
        "prior_recovery": None,
        "replacement_seed_allowed": False,
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    submission_raw = _write_json(
        registry / "submissions" / "array-900.json",
        submission,
    )
    submission_sha = hashlib.sha256(submission_raw).hexdigest()
    scheduler_registry = {
        "schema_version": "prorm-phase2-campaign-scheduler-terminal/v1",
        "status": "terminal_non_success_no_retry",
        "seed": 20260901,
        "attempt_index": 1,
        "phase2_design_sha256": design,
        "base_config_hash": base,
        "phase2_runtime_contract_sha256": runtime,
        "git_commit": commit,
        "accepted_freeze_aggregate_sha256": freeze,
        "cluster_name": "hpc4",
        "array_job_id": "900",
        "array_task_id": 0,
        "slurm_job_id": "777",
        "slurm_restart_count": 0,
        "scheduler_state": "NODE_FAIL",
        "exit_code": "1:0",
        "scheduler_raw_evidence_sha256": scheduler_sha,
        "registry_submission_sha256": submission_sha,
        "registry_execution_sha256": None,
        "retry_authorized": False,
        "replacement_seed_allowed": False,
    }
    scheduler_registry_raw = _write_json(
        registry / "scheduler-terminals" / "seed-20260901-attempt-1.json",
        scheduler_registry,
    )
    scheduler_registry_sha = hashlib.sha256(scheduler_registry_raw).hexdigest()
    _write_claim(
        job.parent / "CLAIM",
        design=design,
        base=base,
        freeze=freeze,
        commit=commit,
        submission_sha=submission_sha,
        scheduler_registry_sha=scheduler_registry_sha,
    )
    ledger = {
        "schema_version": "phase2-seed-attempt-ledger/v3",
        "retry_policy": "single_predeclared_attempt_no_retry",
        "replacement_seed_allowed": False,
        "attempts": [
            {
                "attempt_index": 1,
                "cluster_name": "hpc4",
                "array_job_id": "900",
                "array_task_id": 0,
                "slurm_job_id": "777",
                "status": "terminal_failure",
                "final_outcome_reveal_started": False,
                "log_sha256": scheduler_sha,
            }
        ],
    }
    ledger_raw = _write_json(job / "phase2-attempt-ledger.json", ledger)
    attestation = {
        "schema_version": "phase2-scheduler-terminal-attestation/v1",
        "terminal": True,
        "supports_formal_claim": False,
        "seed": 20260901,
        "attempt_index": 1,
        "cluster_name": "hpc4",
        "slurm_job_id": "777",
        "array_job_id": "900",
        "array_task_id": 0,
        "scheduler_state": "NODE_FAIL",
        "exit_code": "1:0",
        "scheduler_evidence_sha256": scheduler_sha,
        "source_config_hash": base,
        "phase2_design_sha256": design,
        "phase2_runtime_contract_sha256": runtime,
        "git_commit": commit,
        "registry_submission_sha256": submission_sha,
        "registry_execution_sha256": None,
        "accepted_freeze_aggregate_sha256": freeze,
        "final_outcome_reveal_started": False,
    }
    attestation_raw = _write_json(
        job / "scheduler-terminal-attestation.json",
        attestation,
    )
    terminal = {
        "schema_version": "phase2-seed-terminal-failure/v2",
        "terminal_status": "failed",
        "terminal": True,
        "supports_formal_claim": False,
        "seed": 20260901,
        "source_config_hash": base,
        "phase2_design_sha256": design,
        "phase2_runtime_contract_sha256": runtime,
        "capture_method": "scheduler_terminal_reconciliation",
        "evidence_availability": {
            "schema_version": "phase2-seed-failure-evidence-availability/v1",
            "run_manifest": {
                "status": "unavailable",
                "reason": "not_published_before_hard_termination",
            },
            "artifact_metadata": {
                "status": "unavailable",
                "reason": "not_produced_before_failure",
            },
            "environment_identity": {
                "status": "unavailable",
                "reason": "not_recoverable_from_scheduler_evidence",
            },
        },
        "failure": {
            "stage": "scheduler_reconciliation",
            "class": "infrastructure",
            "type": "node_failure",
            "message_sha256": "2" * 64,
            "final_outcome_reveal_started": False,
            "scientific_result_published": False,
        },
        "attempt_ledger": ledger,
        "evidence_sha256_by_role": {"scheduler_terminal_attestation": scheduler_sha},
        "seed_replacement_allowed": False,
    }
    terminal_path = job / "phase2-failure-terminal.json"
    terminal_raw = _write_json(terminal_path, terminal)
    marker = {
        "schema_version": "prorm-phase2-scheduler-terminal-status/v1",
        "status": "SCHEDULER_FAILED",
        "terminal": True,
        "supports_formal_claim": False,
        "seed": 20260901,
        "attempt_index": 1,
        "cluster_name": "hpc4",
        "slurm_job_id": "777",
        "array_job_id": "900",
        "array_task_id": 0,
        "phase2_design_sha256": design,
        "base_config_hash": base,
        "phase2_runtime_contract_sha256": runtime,
        "registry_submission_sha256": submission_sha,
        "registry_execution_sha256": None,
        "registry_scheduler_terminal_sha256": scheduler_registry_sha,
        "attempt_claim_sha256": hashlib.sha256((job.parent / "CLAIM").read_bytes()).hexdigest(),
        "outcome_reveal_marker_sha256": "none",
        "final_outcome_reveal_started": False,
        "scheduler_terminal_attestation_sha256": hashlib.sha256(attestation_raw).hexdigest(),
        "scheduler_raw_evidence_sha256": scheduler_sha,
        "attempt_ledger_sha256": hashlib.sha256(ledger_raw).hexdigest(),
        "terminal_manifest_sha256": hashlib.sha256(terminal_raw).hexdigest(),
    }
    _write_json(job / "SCHEDULER_FAILED", marker)
    return terminal_path, [
        str(terminal_path),
        "20260901",
        design,
        base,
        commit,
        image,
        inventory,
    ]


def test_scheduler_terminal_failure_validates_without_fabricated_gpu_evidence(
    tmp_path: Path,
) -> None:
    terminal, arguments = _scheduler_terminal_tree(tmp_path)
    result = subprocess.run(
        ["python", str(VALIDATOR), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert str(terminal.parent / "SCHEDULER_FAILED") in result.stdout


def test_scheduler_terminal_allows_only_safe_hard_kill_marker_residue(
    tmp_path: Path,
) -> None:
    terminal, arguments = _scheduler_terminal_tree(tmp_path)
    residue = terminal.parent.parent / ".CLAIM.tmp.deadbeef"
    residue.write_text("non-authoritative partial marker\n", encoding="utf-8")
    accepted = subprocess.run(
        ["python", str(VALIDATOR), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    residue.unlink()
    (terminal.parent.parent / ".CLAIM.tmp.deadbeef.extra").mkdir()
    rejected = subprocess.run(
        ["python", str(VALIDATOR), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "unexpected or unsafe" in rejected.stderr


def test_terminal_validator_rejects_caller_cherry_picked_attempt_prefix(
    tmp_path: Path,
) -> None:
    terminal, arguments = _scheduler_terminal_tree(tmp_path)
    (terminal.parent.parent.parent / "attempt-2").mkdir()
    result = subprocess.run(
        ["python", str(VALIDATOR), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "missing, extra, or noncontiguous" in result.stderr


def test_terminal_validator_rejects_arbitrary_outcome_boundary_bytes(
    tmp_path: Path,
) -> None:
    terminal, arguments = _scheduler_terminal_tree(tmp_path)
    job = terminal.parent
    outcome = job.parent / "OUTCOME_REVEAL_STARTED"
    outcome.write_text("forged arbitrary bytes\n", encoding="utf-8")
    outcome_sha = hashlib.sha256(outcome.read_bytes()).hexdigest()

    ledger = json.loads((job / "phase2-attempt-ledger.json").read_text(encoding="utf-8"))
    ledger["attempts"][0]["final_outcome_reveal_started"] = True
    ledger_raw = _write_json(job / "phase2-attempt-ledger.json", ledger)
    attestation = json.loads(
        (job / "scheduler-terminal-attestation.json").read_text(encoding="utf-8")
    )
    attestation["final_outcome_reveal_started"] = True
    attestation_raw = _write_json(
        job / "scheduler-terminal-attestation.json",
        attestation,
    )
    terminal_value = json.loads(terminal.read_text(encoding="utf-8"))
    terminal_value["attempt_ledger"] = ledger
    terminal_value["failure"]["final_outcome_reveal_started"] = True
    terminal_raw = _write_json(terminal, terminal_value)
    marker = json.loads((job / "SCHEDULER_FAILED").read_text(encoding="utf-8"))
    marker["final_outcome_reveal_started"] = True
    marker["outcome_reveal_marker_sha256"] = outcome_sha
    marker["attempt_ledger_sha256"] = hashlib.sha256(ledger_raw).hexdigest()
    marker["scheduler_terminal_attestation_sha256"] = hashlib.sha256(attestation_raw).hexdigest()
    marker["terminal_manifest_sha256"] = hashlib.sha256(terminal_raw).hexdigest()
    _write_json(job / "SCHEDULER_FAILED", marker)

    result = subprocess.run(
        ["python", str(VALIDATOR), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "outcome-reveal boundary marker" in result.stderr


def test_failure_terminalizers_are_parseable_when_bash_is_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is unavailable on this host")
    for source in (COMPUTE_TERMINALIZER, SCHEDULER_TERMINALIZER):
        result = subprocess.run(
            [bash, "-n", str(source)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_scheduler_terminal_marker_heredoc_executes_end_to_end(
    tmp_path: Path,
) -> None:
    source = SCHEDULER_TERMINALIZER.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    current: list[str] | None = None
    for line in source:
        if current is None and "<<'PY'" in line:
            current = []
        elif current is not None and line == "PY":
            blocks.append("\n".join(current) + "\n")
            current = None
        elif current is not None:
            current.append(line)
    assert current is None
    marker_block = next(
        block
        for block in blocks
        if 'staging / "SCHEDULER_FAILED"' in block
        and 'staging / "phase2-failure-terminal.json"' in block
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    design = "a" * 64
    base = "b" * 64
    runtime = "c" * 64
    scheduler_sha = "d" * 64
    scheduler_registry_sha = "e" * 64
    submission_sha = "f" * 64
    ledger = {"attempts": [{"attempt_index": 1}]}
    _write_json(staging / "phase2-attempt-ledger.json", ledger)
    _write_json(
        staging / "scheduler-terminal-attestation.json",
        {
            "final_outcome_reveal_started": False,
            "cluster_name": "hpc4",
            "array_job_id": "900",
            "array_task_id": 0,
            "slurm_job_id": "777",
            "registry_submission_sha256": submission_sha,
            "registry_execution_sha256": None,
            "git_commit": "1" * 40,
            "accepted_freeze_aggregate_sha256": "2" * 64,
        },
    )
    _write_json(
        staging / "phase2-failure-terminal.json",
        {
            "schema_version": "phase2-seed-terminal-failure/v2",
            "capture_method": "scheduler_terminal_reconciliation",
            "seed": 20260901,
            "source_config_hash": base,
            "phase2_design_sha256": design,
            "phase2_runtime_contract_sha256": runtime,
            "evidence_sha256_by_role": {"scheduler_terminal_attestation": scheduler_sha},
        },
    )
    result = subprocess.run(
        [
            "python",
            "-c",
            marker_block,
            str(staging),
            design,
            base,
            runtime,
            "20260901",
            "1",
            scheduler_sha,
            scheduler_registry_sha,
            "",
            "none",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    claim = (staging / "CLAIM").read_text(encoding="utf-8")
    assert f"registry_submission_sha256={submission_sha}\n" in claim
    assert "registry_execution_sha256=none\n" in claim
    assert f"registry_scheduler_terminal_sha256={scheduler_registry_sha}\n" in claim
    marker = json.loads((staging / "SCHEDULER_FAILED").read_text(encoding="utf-8"))
    assert marker["registry_submission_sha256"] == submission_sha
    assert marker["registry_execution_sha256"] is None
    assert marker["registry_scheduler_terminal_sha256"] == scheduler_registry_sha
    assert marker["outcome_reveal_marker_sha256"] == "none"


def test_scheduler_registry_heredoc_binds_exact_scheduler_tuple(
    tmp_path: Path,
) -> None:
    source = SCHEDULER_TERMINALIZER.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    current: list[str] | None = None
    for line in source:
        if current is None and "<<'PY'" in line:
            current = []
        elif current is not None and line == "PY":
            blocks.append("\n".join(current) + "\n")
            current = None
        elif current is not None:
            current.append(line)
    registry_block = next(
        block
        for block in blocks
        if 'staging / "scheduler-registry-terminal.json"' in block and "classification_raw" in block
    )

    staging = tmp_path / "staging"
    submissions = tmp_path / "submissions"
    executions = tmp_path / "executions"
    for directory in (staging, submissions, executions):
        directory.mkdir()
    design = "a" * 64
    base = "b" * 64
    runtime = "c" * 64
    commit = "d" * 40
    freeze = "e" * 64
    scheduler = tmp_path / "sacct.raw"
    scheduler.write_text("hpc4|777|900_0|NODE_FAIL|1:0\n", encoding="utf-8")
    scheduler_sha = hashlib.sha256(scheduler.read_bytes()).hexdigest()
    classification = tmp_path / "classification.json"
    _write_json(
        classification,
        {
            "failure_stage": "scheduler_reconciliation",
            "failure_class": "infrastructure",
            "failure_type": "node_failure",
            "failure_message_sha256": "f" * 64,
            "final_outcome_reveal_started": False,
            "evidence_availability": {},
        },
    )
    _write_json(
        submissions / "array-900.json",
        {
            "schema_version": "prorm-phase2-campaign-submission/v1",
            "status": "committed_while_slurm_held",
            "phase2_design_sha256": design,
            "base_config_hash": base,
            "git_commit": commit,
            "accepted_freeze_aggregate_sha256": freeze,
            "submitted_cluster": "hpc4",
            "replacement_seed_allowed": False,
            "entries": [
                {
                    "seed": 20260901,
                    "attempt_index": 1,
                    "array_job_id": "900",
                    "array_task_id": 0,
                }
            ],
        },
    )
    result = subprocess.run(
        [
            "python",
            "-c",
            registry_block,
            str(classification),
            str(scheduler),
            str(staging),
            "20260901",
            "1",
            "900_0",
            "900",
            "0",
            scheduler_sha,
            design,
            base,
            runtime,
            commit,
            "hpc4",
            freeze,
            str(submissions),
            str(executions),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((staging / "scheduler-registry-terminal.json").read_text(encoding="utf-8"))
    attestation = json.loads(
        (staging / "scheduler-terminal-attestation.json").read_text(encoding="utf-8")
    )
    ledger = json.loads((staging / "phase2-attempt-ledger.json").read_text(encoding="utf-8"))
    for value in (record, attestation):
        assert value["cluster_name"] == "hpc4"
        assert value["array_job_id"] == "900"
        assert value["array_task_id"] == 0
        assert value["slurm_job_id"] == "777"
    assert record["phase2_runtime_contract_sha256"] == runtime
    assert record["scheduler_raw_evidence_sha256"] == scheduler_sha
    assert ledger["attempts"][-1]["slurm_job_id"] == "777"

    bad_submission = json.loads((submissions / "array-900.json").read_text(encoding="utf-8"))
    bad_submission["submitted_cluster"] = "forged-cluster"
    _write_json(submissions / "array-900.json", bad_submission)
    bad_staging = tmp_path / "bad-staging"
    bad_staging.mkdir()
    bad_args = list(result.args)
    bad_args[5] = str(bad_staging)
    rejected = subprocess.run(
        bad_args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "held-array registry" in rejected.stderr


def test_terminal_bundle_publication_recovers_after_injected_mid_bundle_kill(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "job-900_0"
    destination.mkdir()
    names = ("evidence.json", "terminal.json", "FAILED")

    def stage() -> Path:
        staging = destination / ".terminal.tmp"
        staging.mkdir()
        for index, name in enumerate(names):
            (staging / name).write_text(f"{index}:{name}\n", encoding="utf-8")
        return staging

    first_staging = stage()
    interrupted_environment = dict(os.environ)
    interrupted_environment["PRORM_PHASE2_TEST_INTERRUPT_AFTER_PUBLICATION"] = "2"
    interrupted = subprocess.run(
        [
            "python",
            str(PUBLISHER),
            str(first_staging),
            str(destination),
            "FAILED",
            *names,
        ],
        cwd=ROOT,
        env=interrupted_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert interrupted.returncode != 0
    assert (destination / "evidence.json").is_file()
    assert (destination / "terminal.json").is_file()
    assert not (destination / "FAILED").exists()

    shutil.rmtree(first_staging)
    second_staging = stage()
    resumed = subprocess.run(
        [
            "python",
            str(PUBLISHER),
            str(second_staging),
            str(destination),
            "FAILED",
            *names,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert not second_staging.exists()
    assert {path.name for path in destination.iterdir()} == set(names)
    assert (destination / "FAILED").read_text(encoding="utf-8") == "2:FAILED\n"


def test_compute_failure_requires_explicit_classification_before_failed_marker() -> None:
    gpu_job = (ROOT / "scripts" / "hpc4" / "phase2_confirmatory.sbatch").read_text(encoding="utf-8")
    terminalizer = COMPUTE_TERMINALIZER.read_text(encoding="utf-8")

    assert 'marker="FAILURE_PENDING"' in gpu_job
    assert 'marker="SUCCESS_SEALED_SYNC_ERROR"' not in gpu_job
    assert 'marker="SUCCESS"' in gpu_job
    assert "failure-classification.json" in terminalizer
    assert '"${job_dir}/FAILURE_PENDING"' in terminalizer
    assert "phase2-failure-manifest" in terminalizer
    assert "printf 'terminal_manifest_sha256=%s\\n'" in terminalizer
    assert "publish_phase2_terminal_bundle.py" in terminalizer
    assert "unclassified_nonzero_exit" not in terminalizer


def test_scheduler_terminal_is_failure_only_and_never_retry_authorizing() -> None:
    terminalizer = SCHEDULER_TERMINALIZER.read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")

    assert "from datetime import datetime, timezone" in terminalizer
    assert "datetime.now(timezone.utc)" in terminalizer
    assert "datetime import UTC" not in terminalizer
    assert '"capture_method": "scheduler_terminal_reconciliation"' in terminalizer
    assert '"status": "terminal_failure"' in terminalizer
    assert '"retry_eligible": False' not in terminalizer
    assert "Cluster|JobIDRaw|JobID|State|ExitCode" in terminalizer
    assert '"cluster_name": cluster_name' in terminalizer
    assert '"slurm_job_id": slurm_job_id' in terminalizer
    assert '[[ "${PRORM_CLUSTER_NAME}" = "hpc4" ]]' in terminalizer
    assert 'submission.get("submitted_cluster") != cluster_name' in terminalizer
    assert (
        "canonical FAILURE_PENDING must be closed by the compute failure terminalizer"
        in terminalizer
    )
    assert 'fsync_file_and_parent "${scheduler_registry_record}"' in terminalizer
    assert 'fsync_file_and_parent "${attempt_parent}/CLAIM"' in terminalizer
    assert 'fsync_tree "${staging}"' in terminalizer
    assert 'fsync_directory "${attempt_parent}"' in terminalizer
    assert r"\.(?:CLAIM|OUTCOME_REVEAL_STARTED)\.tmp\." in validator
    assert "single_predeclared_attempt_no_retry" in validator
    assert "infrastructure_failure_pre_outcome" not in validator
