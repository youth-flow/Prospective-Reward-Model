from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import smart_reward.phase2_post_recovery_control as control

ROOT = Path(__file__).resolve().parents[1]
GPU_HELPER = ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_array_once.py"
CPU_HELPER = ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_aggregate_attempt.py"


class InjectedCrash(RuntimeError):
    pass


def _load_script(path: Path, stem: str) -> Any:
    name = f"_{stem}_{os.urandom(6).hex()}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_json(path: Path, value: object) -> bytes:
    raw = control._canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _write_receipt(path: Path, value: dict[str, str]) -> bytes:
    raw = control._aggregate_receipt_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _cpu_history_row(
    *,
    job_id: str,
    job_name: str,
    state: str,
) -> dict[str, str]:
    success = state == "COMPLETED"
    exit_code = "0:0" if success else "1:0"
    return {
        "JobIDRaw": job_id,
        "JobID": job_id,
        "JobName": job_name,
        "State": state,
        "ExitCode": exit_code,
        "DerivedExitCode": exit_code,
        "Cluster": "hpc4",
        "Account": "sigroup",
        "Partition": "amd",
        "NNodes": "1",
        "NCPUS": "4",
        "Submit": "2026-07-25T00:00:00",
        "Timelimit": "01:00:00",
        "ReqTRES": "billing=4,cpu=4,mem=16G,node=1",
        "AllocTRES": "billing=4,cpu=4,mem=16G,node=1",
    }


class Scheduler:
    def __init__(
        self,
        *,
        cpu_fields: tuple[str, ...],
        job_name: str,
        script_bytes: bytes,
    ) -> None:
        self.cpu_fields = cpu_fields
        self.job_name = job_name
        self.script_bytes = script_bytes
        self.rows: dict[str, dict[str, str]] = {
            "2999": _cpu_history_row(
                job_id="2999",
                job_name=job_name,
                state="FAILED",
            ),
            "3000": _cpu_history_row(
                job_id="3000",
                job_name=job_name,
                state="COMPLETED",
            ),
        }
        self.live_ids: list[str] = []
        self.terminal_error = False
        self.terminal_raw = (
            b"3000|3000|COMPLETED|0:0|0:0|hpc4|sigroup|amd|1|4|"
            b"billing=4,cpu=4,mem=16G,node=1|"
            b"billing=4,cpu=4,mem=16G,node=1\n"
        )
        self.fail_all = False
        self.calls: list[tuple[str, ...]] = []

    def history_raw(self) -> str:
        return "".join(
            "|".join(row[field] for field in self.cpu_fields) + "\n"
            for _, row in sorted(self.rows.items(), key=lambda item: int(item[0]))
        )

    def __call__(self, arguments: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        args = tuple(os.fspath(item) for item in arguments)
        self.calls.append(args)
        if self.fail_all:
            raise AssertionError("completed publication consulted scheduler state")
        text = bool(kwargs.get("text", False))
        if args[0] == "git":
            assert not text
            return subprocess.CompletedProcess(args, 0, self.script_bytes, b"")
        if args[0] == "squeue":
            assert f"--name={self.job_name}" in args
            stdout = "".join(f"{job_id}\n" for job_id in self.live_ids)
            return subprocess.CompletedProcess(args, 0, stdout, "")
        if args[0] == "sacct" and any(item.startswith("--name=") for item in args):
            stdout = self.history_raw()
            return subprocess.CompletedProcess(args, 0, stdout, "")
        if args[0] == "sacct" and "-j" in args:
            if self.terminal_error:
                return subprocess.CompletedProcess(args, 1, b"", b"sacct failed")
            assert not text
            return subprocess.CompletedProcess(args, 0, self.terminal_raw, b"")
        raise AssertionError(f"unexpected scheduler query: {args!r}")


@dataclass
class BuiltAttempt:
    project: Path
    repository: Path
    aggregate: Path
    attempt_root: Path
    campaign_root: Path
    scheduler: Scheduler
    cpu_intent_sha256: str
    cpu_attempt_sha256: str

    @property
    def final_evidence(self) -> Path:
        return Path(f"{self.aggregate}.evidence")

    def capture(self) -> dict[str, object]:
        return control.capture_post_recovery_aggregate_terminal_evidence(
            self.aggregate,
            attempt_job_id="3000",
        )

    def verify(self) -> dict[str, object]:
        return control.verify_post_recovery_aggregate_success_receipt(self.aggregate)


@pytest.fixture
def built_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> BuiltAttempt:
    if os.name == "nt":
        short_root = tmp_path.parent / f"p2-{os.urandom(4).hex()}"
        short_root.mkdir()
        tmp_path = short_root
    gpu = _load_script(GPU_HELPER, "gpu_submit")
    cpu = _load_script(CPU_HELPER, "cpu_submit")
    project = tmp_path / "project"
    repository = tmp_path / "repository"
    aggregates = project / "aggregates"
    aggregates.mkdir(parents=True)
    repository.mkdir()
    aggregate = aggregates / "phase2-post-recovery-calibration-aggregate.json"
    campaign_root = project / "runs" / "phase2-post-recovery-aggregate-attempts" / aggregate.name
    live_registry = campaign_root / "submission-registry"
    attempt_root = campaign_root / "job-3000"
    attempt_evidence = attempt_root / "evidence"

    design_sha = "a" * 64
    base_config_hash = "b" * 64
    authorization_sha = "d" * 64
    producer_commit = "e" * 40
    aggregator_commit = "f" * 40
    image_sha = "1" * 64
    inventory_sha = "2" * 64
    overlay_relative = "configs/common_beta_post_recovery_calibration.yaml"
    base_relative = "configs/common_beta_pilot_base.yaml"
    overlay_file = attempt_evidence.joinpath(*overlay_relative.split("/"))
    base_file = attempt_evidence.joinpath(*base_relative.split("/"))
    overlay_file.parent.mkdir(parents=True)
    overlay_file.write_bytes(b"schema_version: fixture-overlay\n")
    base_file.write_bytes(b"schema_version: fixture-base\n")
    overlay_sha = _sha(overlay_file)
    base_sha = _sha(base_file)

    gpu_script_relative = "scripts/hpc4/phase2_post_recovery_calibration.sbatch"
    gpu_script = repository.joinpath(*gpu_script_relative.split("/"))
    gpu_script.parent.mkdir(parents=True)
    gpu_script.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
    hf_cache = tmp_path / "hf-cache"
    gpu_exports = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PRORM_PROJECT_ROOT": os.fspath(project),
        "PRORM_SCRATCH_ROOT": os.fspath(tmp_path / "scratch"),
        "PRORM_REPO_ROOT": os.fspath(repository),
        "PRORM_IMAGE": os.fspath(tmp_path / "image.sif"),
        "PRORM_IMAGE_SHA256": image_sha,
        "PRORM_HF_CACHE": os.fspath(hf_cache),
        "PRORM_HF_INVENTORY": (f"{hf_cache}/inventories/{base_config_hash}.json"),
        "PRORM_HF_INVENTORY_SHA256": inventory_sha,
        "PRORM_RECOVERY_AUTHORIZATION": os.fspath(tmp_path / "authorization.json"),
        "PRORM_POST_RECOVERY_OVERLAY_REL": overlay_relative,
        "PRORM_PHASE2_BASE_REL": base_relative,
        "PRORM_POST_RECOVERY_OVERLAY_SHA256": overlay_sha,
        "PRORM_PHASE2_BASE_SHA256": base_sha,
        "PRORM_POST_RECOVERY_DESIGN_SHA256": design_sha,
        "PRORM_PHASE2_BASE_CONFIG_HASH": base_config_hash,
        "PRORM_RECOVERY_AUTHORIZATION_SHA256": authorization_sha,
        "PRORM_OPTIMIZER_SCHEDULE_SHA256": control.OPTIMIZER_SCHEDULE_SHA256,
        "PRORM_GIT_COMMIT": producer_commit,
        "PRORM_POST_RECOVERY_PILOT_PHASE": "calibration",
        "PRORM_POST_RECOVERY_NAMESPACE": "calibration",
        "PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT": "0",
        "PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT": "0",
    }
    gpu_export_spec = ",".join(f"{key}={value}" for key, value in gpu_exports.items())
    gpu_job_name = f"prorm-p2-post-calibration-{design_sha[:12]}"
    gpu_intent = gpu._intent_payload(
        pilot_phase="calibration",
        design_sha256=design_sha,
        base_config_hash=base_config_hash,
        authorization_sha256=authorization_sha,
        optimizer_schedule_sha256=control.OPTIMIZER_SCHEDULE_SHA256,
        git_commit=producer_commit,
        image_sha256=image_sha,
        inventory_sha256=inventory_sha,
        overlay_sha256=overlay_sha,
        base_file_sha256=base_sha,
        sbatch_script_relative=gpu_script_relative,
        sbatch_script_sha256=_sha(gpu_script),
        export_spec=gpu_export_spec,
        export_spec_sha256=_sha_bytes(gpu_export_spec.encode()),
        walltime="12:00:00",
        job_name=gpu_job_name,
        project_root=os.fspath(project),
        repository_root=os.fspath(repository),
        submitter_user="researcher",
        created_at_utc="2026-07-25T00:00:00Z",
    )
    gpu_intent_raw = control._canonical_json(gpu_intent)
    gpu_intent_sha = _sha_bytes(gpu_intent_raw)
    gpu_scontrol_raw = (
        "ArrayJobId=2000 ArrayTaskId=0-2%2 "
        f"JobName={gpu_job_name} UserId=researcher(1000) "
        "Account=sigroup Partition=gpu-l20 QOS=l20_qos "
        "Requeue=0 Restarts=0 ArrayTaskThrottle=2 "
        "NumNodes=1 NumTasks=1 NumCPUs=8 CPUs/Task=8 "
        "MinMemoryNode=96G TimeLimit=12:00:00 "
        "TRES=cpu=8,mem=96G,node=1,gres/gpu=1 "
        "TresPerNode=gres/gpu:1 "
        f"Command={gpu_script} WorkDir={repository} "
        "JobState=PENDING Reason=JobHeldUser\n"
    )
    state, gpu_scheduler = gpu._parse_scontrol_records(
        gpu_scontrol_raw,
        array_job_id="2000",
        expected_name=gpu_job_name,
        expected_walltime="12:00:00",
        expected_command=gpu_script,
        expected_workdir=repository,
        expected_user="researcher",
    )
    assert state == "HELD" and gpu_scheduler is not None
    gpu_submission = gpu._submission_payload(
        intent=gpu_intent,
        intent_sha256=gpu_intent_sha,
        array_job_id="2000",
        submitted_cluster="hpc4",
        scheduler_request=gpu_scheduler,
    )
    gpu_submission_raw = control._canonical_json(gpu_submission)
    gpu_submission_sha = _sha_bytes(gpu_submission_raw)
    gpu_registry = attempt_evidence / "submission-registry"
    gpu_registry.mkdir(parents=True)
    (gpu_registry / "intent.json").write_bytes(gpu_intent_raw)
    (gpu_registry / "submission.json").write_bytes(gpu_submission_raw)

    cpu_script_relative = "scripts/hpc4/phase2_post_recovery_aggregate.sbatch"
    cpu_script = repository.joinpath(*cpu_script_relative.split("/"))
    cpu_script.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
    cpu_script_raw = cpu_script.read_bytes()
    cpu_job_name = f"prorm-p2-post-agg-{design_sha[:12]}-fixture"
    cpu_export_spec = ",".join(
        (
            f"PRORM_PROJECT_ROOT={project}",
            f"PRORM_REPO_ROOT={repository}",
            f"PRORM_POST_RECOVERY_DESIGN_SHA256={design_sha}",
            "PRORM_POST_RECOVERY_ARRAY_JOB_ID=2000",
            f"PRORM_AGGREGATOR_GIT_COMMIT={aggregator_commit}",
            f"PRORM_POST_RECOVERY_AGGREGATE_OUTPUT={aggregate}",
            "PRORM_POST_RECOVERY_PILOT_PHASE=calibration",
        )
    )
    cpu_intent = cpu._intent_payload(
        pilot_phase="calibration",
        design_sha256=design_sha,
        pilot_array_job_id="2000",
        aggregator_git_commit=aggregator_commit,
        project_root=project,
        repository_root=repository,
        output=aggregate,
        partition="amd",
        walltime="01:00:00",
        workload_export_spec=cpu_export_spec,
        script_relative=cpu_script_relative,
        script_sha256=_sha_bytes(cpu_script_raw),
        script_git_blob_sha1=cpu._git_blob_sha1(cpu_script_raw),
        script_size_bytes=len(cpu_script_raw),
        submitter_user="researcher",
        job_name=cpu_job_name,
        created_at_utc="2026-07-25T00:00:00Z",
    )
    cpu_intent_raw = control._canonical_json(cpu_intent)
    cpu_intent_sha = _sha_bytes(cpu_intent_raw)
    workload_sha = _sha_bytes(cpu_export_spec.encode())
    aggregate_submission = attempt_evidence / "aggregate-submission"
    for directory in (
        aggregate_submission / "attempts",
        aggregate_submission / "failures",
        aggregate_submission / cpu.CONTROLLER_READBACK_DIRECTORY,
        live_registry / "attempts",
        live_registry / "failures",
        live_registry / cpu.CONTROLLER_READBACK_DIRECTORY,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for root in (aggregate_submission, live_registry):
        (root / cpu.SCRIPT_EVIDENCE_FILENAME).write_bytes(cpu_script_raw)

    def cpu_attempt(index: int, job_id: str) -> tuple[dict[str, object], bytes]:
        scheduler_export = (
            f"{cpu_export_spec}"
            f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={live_registry}"
            f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={cpu_intent_sha}"
            f",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX={index}"
            ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
            f"{workload_sha}"
        )
        comment = f"prorm-aggregate:{cpu_intent_sha}:attempt-{index}"
        raw = (
            f"JobId={job_id} JobName={cpu_job_name} "
            "UserId=researcher(1000) Account=sigroup Partition=amd "
            "Requeue=0 Restarts=0 NumNodes=1 NumTasks=1 NumCPUs=4 "
            "CPUs/Task=4 MinMemoryNode=16G TimeLimit=01:00:00 "
            f"Command=(null) WorkDir={repository} Comment={comment} "
            "TRES=cpu=4,mem=16G,node=1 JobState=PENDING Reason=JobHeldUser "
            "BatchFlag=1\n"
        )
        parsed_state, scheduler = cpu._parse_scontrol(
            raw,
            job_id=job_id,
            intent=cpu_intent,
            intent_sha256=cpu_intent_sha,
            attempt_index=index,
            repository_root=repository,
        )
        assert parsed_state == "HELD" and scheduler is not None
        relative = cpu._controller_readback_relative(index)
        for root in (aggregate_submission, live_registry):
            root.joinpath(*relative.split("/")).write_bytes(cpu_script_raw)
        submission_command = cpu._sbatch_command(
            intent=cpu_intent,
            intent_sha256=cpu_intent_sha,
            attempt_index=index,
            scheduler_export_spec=scheduler_export,
            repository_root=repository,
            log_root=(project / "slurm-logs" / "phase2-post-recovery-aggregate" / aggregate.name),
        )
        batch_script = {
            "schema_version": cpu.SCRIPT_BINDING_SCHEMA,
            "transport": cpu.SCRIPT_TRANSPORT,
            "submission_command": list(submission_command),
            "stdin_sha256": _sha_bytes(cpu_script_raw),
            "stdin_size_bytes": len(cpu_script_raw),
            "controller_readback": {
                "query": list(cpu._controller_readback_query(job_id)),
                "relative_path": relative,
                "sha256": _sha_bytes(cpu_script_raw),
                "size_bytes": len(cpu_script_raw),
            },
            "controller_matches_committed": True,
        }
        value = cpu._attempt_payload(
            intent=cpu_intent,
            intent_sha256=cpu_intent_sha,
            attempt_index=index,
            job_id=job_id,
            scheduler_export_spec=scheduler_export,
            scheduler_request=scheduler,
            batch_script=batch_script,
        )
        return value, control._canonical_json(value)

    attempt_one, attempt_one_raw = cpu_attempt(1, "2999")
    attempt_two, attempt_two_raw = cpu_attempt(2, "3000")
    attempt_one_sha = _sha_bytes(attempt_one_raw)
    attempt_two_sha = _sha_bytes(attempt_two_raw)
    failure_row = _cpu_history_row(
        job_id="2999",
        job_name=cpu_job_name,
        state="FAILED",
    )
    failure_raw = ("|".join(failure_row[field] for field in cpu._SACCT_FIELDS) + "\n").encode()
    failure = cpu._failure_payload(
        intent_sha256=cpu_intent_sha,
        attempt_index=1,
        job_id="2999",
        attempt_ledger_sha256=attempt_one_sha,
        row=failure_row,
        query=cpu._failure_sacct_query("2999"),
        raw_filename="job-2999.sacct.psv",
        raw_sha256=_sha_bytes(failure_raw),
        raw_size_bytes=len(failure_raw),
    )
    failure_raw_json = control._canonical_json(failure)
    failure_sha = _sha_bytes(failure_raw_json)
    failure_chain = [
        {
            "attempt_index": 1,
            "slurm_job_id": "2999",
            "attempt_ledger_sha256": attempt_one_sha,
            "filename": "job-2999.json",
            "sha256": failure_sha,
        }
    ]
    failure_chain_raw = control._canonical_json(failure_chain)
    failure_chain_sha = _sha_bytes(failure_chain_raw)

    for path in (
        aggregate_submission / "intent.json",
        live_registry / "intent.json",
    ):
        path.write_bytes(cpu_intent_raw)
    (aggregate_submission / "attempt.json").write_bytes(attempt_two_raw)
    (aggregate_submission / "failure-chain.json").write_bytes(failure_chain_raw)
    for root in (aggregate_submission, live_registry):
        (root / "attempts" / "attempt-0001.json").write_bytes(attempt_one_raw)
        (root / "attempts" / "attempt-0002.json").write_bytes(attempt_two_raw)
        (root / "failures" / "job-2999.json").write_bytes(failure_raw_json)
        (root / "failures" / "job-2999.sacct.psv").write_bytes(failure_raw)

    final_prefix = f"{aggregate.name}.evidence"
    aggregate_value = {
        "schema_version": "common-beta-pilot-selection-aggregate/v3",
        "pilot_phase": "calibration",
        "phase2_design_sha256": design_sha,
        "source_config_hash": base_config_hash,
        "post_recovery_control": {
            "schema_version": "phase2-post-recovery-aggregation-control/v1",
            "pilot_phase": "calibration",
            "phase2_overlay": f"{final_prefix}/{overlay_relative}",
            "phase2_overlay_repo_relative": overlay_relative,
            "phase2_overlay_sha256": overlay_sha,
            "phase2_overlay_git_blob_sha1": "3" * 40,
            "phase2_overlay_git_commit": producer_commit,
            "normalized_phase2_config": {"design": {"source_config": base_relative}},
            "normalized_phase2_config_sha256": "4" * 64,
            "recovery_authorization": "authorization.json",
            "recovery_authorization_sha256": authorization_sha,
            "optimizer_schedule_sha256": control.OPTIMIZER_SCHEDULE_SHA256,
            "submission_intent": (f"{final_prefix}/submission-registry/intent.json"),
            "submission_intent_sha256": gpu_intent_sha,
            "submission_ledger": (f"{final_prefix}/submission-registry/submission.json"),
            "submission_ledger_sha256": gpu_submission_sha,
            "pilot_terminal_evidence": "terminal.json",
            "pilot_terminal_evidence_sha256": "5" * 64,
            "pilot_array_job_id": "2000",
            "ordered_seeds": [20260801, 20260802, 20260803],
            "materialization_mode": "fresh",
            "recovery_outputs_reused": False,
            "all_tasks_terminal_completed_zero_exit": True,
            "post_recovery_validator_source_sha256": "6" * 64,
            "phase2_deep_validator_source_sha256": "7" * 64,
        },
        "aggregation_identity": {
            "schema_version": "phase2-aggregation-identity/v1",
            "aggregator_git_commit": aggregator_commit,
            "producer_git_commit": producer_commit,
            "image_sha256": image_sha,
            "hf_inventory_sha256": inventory_sha,
            "validator_source_sha256": "8" * 64,
        },
    }
    attempt_root.mkdir(parents=True, exist_ok=True)
    staged_aggregate = attempt_root / "aggregate.json"
    _write_json(staged_aggregate, aggregate_value)
    ready = {
        "schema_version": control.POST_RECOVERY_AGGREGATE_ATTEMPT_READY_SCHEMA,
        "status": "READY",
        "slurm_job_id": "3000",
        "slurm_job_is_array": "false",
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "amd",
        "restart_count": "0",
        "pilot_array_job_id": "2000",
        "pilot_phase": "calibration",
        "phase2_design_sha256": design_sha,
        "base_config_hash": base_config_hash,
        "recovery_authorization_sha256": authorization_sha,
        "optimizer_schedule_sha256": control.OPTIMIZER_SCHEDULE_SHA256,
        "pilot_terminal_evidence_sha256": "5" * 64,
        "submission_intent_sha256": gpu_intent_sha,
        "submission_ledger_sha256": gpu_submission_sha,
        "aggregate_submission_intent_sha256": cpu_intent_sha,
        "aggregate_submission_attempt_sha256": attempt_two_sha,
        "aggregate_submission_attempt_index": "2",
        "aggregate_submission_failure_chain_sha256": failure_chain_sha,
        "phase2_overlay_sha256": overlay_sha,
        "phase2_base_sha256": base_sha,
        "aggregator_git_commit": aggregator_commit,
        "producer_git_commit": producer_commit,
        "image_sha256": image_sha,
        "hf_inventory_sha256": inventory_sha,
        "final_output": os.fspath(aggregate),
        "final_evidence_root": f"{aggregate}.evidence",
        "attempt_aggregate": "aggregate.json",
        "attempt_evidence": "evidence",
        "aggregate_sha256": _sha(staged_aggregate),
        "final_namespace_untouched": "true",
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    _write_receipt(attempt_root / "READY", ready)

    scheduler = Scheduler(
        cpu_fields=tuple(control._AGGREGATE_SUBMIT_SACCT_FIELDS),
        job_name=cpu_job_name,
        script_bytes=cpu_script.read_bytes(),
    )
    monkeypatch.setattr(control.subprocess, "run", scheduler)
    # Directory fsync is not uniformly available on Windows CI.
    if os.name == "nt":
        monkeypatch.setattr(control, "_fsync_directory", lambda _path: None)
        monkeypatch.setattr(control, "_fsync_tree", lambda _path: None)

    built = BuiltAttempt(
        project=project,
        repository=repository,
        aggregate=aggregate,
        attempt_root=attempt_root,
        campaign_root=campaign_root,
        scheduler=scheduler,
        cpu_intent_sha256=cpu_intent_sha,
        cpu_attempt_sha256=attempt_two_sha,
    )
    verified = control.verify_post_recovery_aggregate_attempt_ready(
        aggregate,
        attempt_job_id="3000",
    )
    assert verified["aggregate_sha256"] == _sha(staged_aggregate)
    return built


def _post_file_install_crash(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
) -> None:
    original = control._write_exclusive
    fired = False

    def wrapped(path: Path, raw: bytes, *, name: str) -> None:
        nonlocal fired
        original(path, raw, name=name)
        if name == target_name and not fired:
            fired = True
            assert path.is_file()
            raise InjectedCrash(target_name)

    target_name = name
    monkeypatch.setattr(control, "_write_exclusive", wrapped)


def _assert_no_consumable_aggregate(built: BuiltAttempt) -> None:
    for suffix in (
        "",
        ".PUBLISHED",
        ".TERMINAL.sacct.psv",
        ".TERMINAL.json",
        ".SUCCESS",
    ):
        assert not Path(f"{built.aggregate}{suffix}").exists()


def _evidence_relative(built: BuiltAttempt, path: Path) -> str | None:
    try:
        return path.absolute().relative_to(built.final_evidence.absolute()).as_posix()
    except ValueError:
        return None


def _evidence_claim_relative() -> str:
    return f"aggregation-attempt/{control.POST_RECOVERY_AGGREGATE_EVIDENCE_CLAIM}"


def _crash_after_evidence_file(
    built: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    *,
    relative: str,
) -> None:
    original = control._install_evidence_file_noreplace
    fired = False

    def installed_then_crash(
        source: Path,
        destination: Path,
        *,
        expected: dict[str, object],
        name: str,
    ) -> bool:
        nonlocal fired
        installed = original(
            source,
            destination,
            expected=expected,
            name=name,
        )
        if _evidence_relative(built, destination) == relative and not fired:
            fired = True
            assert destination.is_file()
            raise InjectedCrash(f"after-evidence-file:{relative}")
        return installed

    monkeypatch.setattr(
        control,
        "_install_evidence_file_noreplace",
        installed_then_crash,
    )


def _crash_after_publication_owner(
    built: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _post_file_install_crash(
        monkeypatch,
        name="post-recovery aggregate ATTEMPT receipt",
    )
    with pytest.raises(InjectedCrash, match="ATTEMPT"):
        built.capture()
    assert Path(f"{built.aggregate}.ATTEMPT").is_file()
    assert not built.final_evidence.exists()


def _force_write(path: Path, raw: bytes) -> None:
    if os.name != "nt":
        os.chmod(path, 0o640)
    path.write_bytes(raw)


def test_evidence_publication_has_no_directory_rename_or_mv_dependency() -> None:
    source_path = Path(control.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    for forbidden in (
        "renameat2",
        "_rename_directory_noreplace",
        "mv --no-clobber",
    ):
        assert forbidden not in source


@pytest.mark.parametrize(
    ("phase", "write_name", "installed_suffix"),
    [
        ("owner", "post-recovery aggregate ATTEMPT receipt", ".ATTEMPT"),
        ("aggregate", "post-recovery aggregate", ""),
        ("publication", "post-recovery aggregate PUBLISHED receipt", ".PUBLISHED"),
        ("raw", "raw aggregation sacct evidence", ".TERMINAL.sacct.psv"),
        (
            "terminal",
            "post-recovery aggregate terminal evidence",
            ".TERMINAL.json",
        ),
        ("success", "post-recovery aggregate SUCCESS receipt", ".SUCCESS"),
    ],
)
def test_file_phase_post_install_crash_resumes_to_one_exact_success(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    write_name: str,
    installed_suffix: str,
) -> None:
    _post_file_install_crash(monkeypatch, name=write_name)
    with pytest.raises(InjectedCrash, match=write_name):
        built_attempt.capture()
    installed = (
        built_attempt.aggregate
        if installed_suffix == ""
        else Path(f"{built_attempt.aggregate}{installed_suffix}")
    )
    assert installed.exists(), phase

    resumed = built_attempt.capture()
    verified = built_attempt.verify()
    assert resumed["receipt_sha256"] == verified["receipt_sha256"]
    assert Path(f"{built_attempt.aggregate}.SUCCESS").is_file()


def test_evidence_root_post_mkdir_crash_resumes_exactly(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = control._ensure_evidence_directory
    fired = False

    def created_then_crash(path: Path, *, name: str) -> bool:
        nonlocal fired
        created = original(path, name=name)
        if path.absolute() == built_attempt.final_evidence.absolute() and not fired:
            fired = True
            assert created is True
            assert path.is_dir()
            raise InjectedCrash("after-evidence-root-mkdir")
        return created

    monkeypatch.setattr(
        control,
        "_ensure_evidence_directory",
        created_then_crash,
    )
    with pytest.raises(InjectedCrash, match="after-evidence-root-mkdir"):
        built_attempt.capture()
    assert built_attempt.final_evidence.is_dir()
    assert control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="post-mkdir evidence root",
    ) == {
        "schema_version": "prorm-exact-evidence-tree-manifest/v1",
        "directories": [],
        "files": {},
    }
    _assert_no_consumable_aggregate(built_attempt)

    built_attempt.capture()
    built_attempt.verify()


def test_evidence_root_pre_mkdir_crash_has_no_visible_target_then_resumes(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = control._ensure_evidence_directory
    fired = False

    def crash_before_create(path: Path, *, name: str) -> bool:
        nonlocal fired
        if path.absolute() == built_attempt.final_evidence.absolute() and not fired:
            fired = True
            assert not path.exists()
            raise InjectedCrash("before-evidence-root-mkdir")
        return original(path, name=name)

    monkeypatch.setattr(control, "_ensure_evidence_directory", crash_before_create)
    with pytest.raises(InjectedCrash, match="before-evidence-root-mkdir"):
        built_attempt.capture()
    assert not built_attempt.final_evidence.exists()
    assert list(
        built_attempt.aggregate.parent.glob(f".{built_attempt.final_evidence.name}.publishing-*")
    )

    built_attempt.capture()
    built_attempt.verify()


def test_evidence_claim_directory_post_mkdir_crash_resumes_exactly(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = control._ensure_evidence_directory
    claim_parent = built_attempt.final_evidence / "aggregation-attempt"
    fired = False

    def created_then_crash(path: Path, *, name: str) -> bool:
        nonlocal fired
        created = original(path, name=name)
        if path.absolute() == claim_parent.absolute() and not fired:
            fired = True
            assert created is True
            raise InjectedCrash("after-evidence-claim-directory-mkdir")
        return created

    monkeypatch.setattr(control, "_ensure_evidence_directory", created_then_crash)
    with pytest.raises(InjectedCrash, match="claim-directory"):
        built_attempt.capture()
    tree = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="pre-claim evidence root",
    )
    assert tree["directories"] == ["aggregation-attempt"]
    assert tree["files"] == {}
    _assert_no_consumable_aggregate(built_attempt)

    built_attempt.capture()
    built_attempt.verify()


def test_evidence_claim_post_install_crash_resumes_without_rewriting_claim(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_relative = _evidence_claim_relative()
    _crash_after_evidence_file(
        built_attempt,
        monkeypatch,
        relative=claim_relative,
    )
    with pytest.raises(InjectedCrash, match="after-evidence-file"):
        built_attempt.capture()
    claim_path = built_attempt.final_evidence.joinpath(*claim_relative.split("/"))
    before = claim_path.read_bytes()
    before_inode = claim_path.stat().st_ino
    tree = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="claim-only evidence root",
    )
    assert set(tree["files"]) == {claim_relative}
    _assert_no_consumable_aggregate(built_attempt)

    built_attempt.capture()
    assert claim_path.read_bytes() == before
    assert claim_path.stat().st_ino == before_inode
    built_attempt.verify()


def test_complete_evidence_post_return_crash_resumes_without_rewriting_tree(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = control._publish_attempt_evidence
    fired = False

    def published_then_crash(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal fired
        result = original(*args, **kwargs)
        if not fired:
            fired = True
            raise InjectedCrash("after-evidence-publisher-return")
        return result

    monkeypatch.setattr(control, "_publish_attempt_evidence", published_then_crash)
    with pytest.raises(InjectedCrash, match="publisher-return"):
        built_attempt.capture()
    before = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="complete pre-aggregate evidence tree",
    )
    claim_path = built_attempt.final_evidence.joinpath(*_evidence_claim_relative().split("/"))
    claim_inode = claim_path.stat().st_ino
    _assert_no_consumable_aggregate(built_attempt)

    built_attempt.capture()
    after = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="complete published evidence tree",
    )
    assert after == before
    assert claim_path.stat().st_ino == claim_inode
    built_attempt.verify()


@pytest.mark.parametrize(
    "relative",
    [
        "aggregation-attempt/READY",
        "aggregation-attempt/AUTHORITY.json",
        "aggregate-submission/attempts/attempt-0001.json",
        "aggregate-submission/controller/attempt-0002.sbatch",
        "configs/common_beta_post_recovery_calibration.yaml",
        "submission-registry/submission.json",
    ],
)
def test_each_representative_incremental_file_boundary_resumes_without_rewrite(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    _crash_after_evidence_file(
        built_attempt,
        monkeypatch,
        relative=relative,
    )
    with pytest.raises(InjectedCrash, match="after-evidence-file"):
        built_attempt.capture()
    installed = built_attempt.final_evidence.joinpath(*relative.split("/"))
    before = installed.read_bytes()
    before_inode = installed.stat().st_ino
    _assert_no_consumable_aggregate(built_attempt)

    built_attempt.capture()
    assert installed.read_bytes() == before
    assert installed.stat().st_ino == before_inode
    built_attempt.verify()


def test_authority_bytes_are_deterministic_across_resume_time(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "aggregation-attempt/AUTHORITY.json"
    _crash_after_evidence_file(
        built_attempt,
        monkeypatch,
        relative=relative,
    )
    with pytest.raises(InjectedCrash):
        built_attempt.capture()
    authority_path = built_attempt.final_evidence.joinpath(*relative.split("/"))
    before = authority_path.read_bytes()
    before_inode = authority_path.stat().st_ino
    owner = control.parse_post_recovery_aggregate_publication_owner(
        Path(f"{built_attempt.aggregate}.ATTEMPT")
    )
    assert json.loads(before)["captured_at_utc"] == owner["created_at_utc"]

    monkeypatch.setattr(control, "_utc_now", lambda: "2036-01-02T03:04:05Z")
    built_attempt.capture()
    assert authority_path.read_bytes() == before
    assert authority_path.stat().st_ino == before_inode
    built_attempt.verify()


def test_authority_verifier_requires_outer_attempt_sealing_time(
    built_attempt: BuiltAttempt,
) -> None:
    built_attempt.capture()
    evidence_root = built_attempt.final_evidence
    ready_path = evidence_root / "aggregation-attempt" / "READY"
    ready = control.parse_post_recovery_aggregate_attempt_ready(ready_path)
    owner = control.parse_post_recovery_aggregate_publication_owner(
        Path(f"{built_attempt.aggregate}.ATTEMPT")
    )
    submission = control._verify_aggregate_submission_bundle(
        attempt_evidence=evidence_root,
        ready=ready,
        aggregate_file=built_attempt.aggregate,
        project_root=built_attempt.project,
        require_live_registry=False,
    )

    verified = control._verify_aggregate_submission_authority_evidence(
        evidence_root,
        ready_sha256=_sha(ready_path),
        submission=submission,
        expected_captured_at_utc=owner["created_at_utc"],
        fresh_authority=None,
    )
    assert verified["payload"]["captured_at_utc"] == owner["created_at_utc"]
    with pytest.raises(ValueError, match="authority identity is invalid"):
        control._verify_aggregate_submission_authority_evidence(
            evidence_root,
            ready_sha256=_sha(ready_path),
            submission=submission,
            expected_captured_at_utc="2036-01-02T03:04:05Z",
            fresh_authority=None,
        )


def test_exact_tree_walk_error_fails_closed(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied_walk(
        *_args: object,
        onerror: object = None,
        **_kwargs: object,
    ) -> object:
        assert callable(onerror)
        onerror(PermissionError("synthetic traversal denial"))
        if False:
            yield None

    monkeypatch.setattr(control.os, "walk", denied_walk)
    with pytest.raises(ValueError, match="cannot be traversed exactly"):
        control._directory_tree_manifest(
            built_attempt.attempt_root / "evidence",
            name="walk-denied aggregate evidence",
        )


@pytest.mark.parametrize(
    "foreign_kind",
    ["hidden-file", "expected-directory", "ready-without-claim"],
)
def test_preclaim_foreign_or_payload_tree_is_preserved_and_rejected(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    foreign_kind: str,
) -> None:
    _crash_after_publication_owner(built_attempt, monkeypatch)
    built_attempt.final_evidence.mkdir()
    if foreign_kind == "hidden-file":
        (built_attempt.final_evidence / ".foreign").write_bytes(b"foreign")
    elif foreign_kind == "expected-directory":
        (built_attempt.final_evidence / "configs").mkdir()
    else:
        claim_parent = built_attempt.final_evidence / "aggregation-attempt"
        claim_parent.mkdir()
        (claim_parent / "READY").write_bytes((built_attempt.attempt_root / "READY").read_bytes())
    before = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="foreign preclaim evidence root",
    )

    with pytest.raises(ValueError):
        built_attempt.capture()
    after = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="foreign preclaim evidence root",
    )
    assert after == before
    _assert_no_consumable_aggregate(built_attempt)


@pytest.mark.parametrize(
    "mutation",
    ["claim", "payload-same-size", "hidden-file", "extra-directory"],
)
def test_claimed_partial_tree_tampering_or_extra_entry_fails_without_progress(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    crash_relative = (
        "aggregation-attempt/READY"
        if mutation == "payload-same-size"
        else _evidence_claim_relative()
    )
    _crash_after_evidence_file(
        built_attempt,
        monkeypatch,
        relative=crash_relative,
    )
    with pytest.raises(InjectedCrash):
        built_attempt.capture()

    if mutation == "claim":
        target = built_attempt.final_evidence.joinpath(*_evidence_claim_relative().split("/"))
        raw = target.read_bytes()
        _force_write(target, raw[:-1] + bytes([raw[-1] ^ 1]))
    elif mutation == "payload-same-size":
        target = built_attempt.final_evidence / "aggregation-attempt" / "READY"
        raw = target.read_bytes()
        _force_write(target, bytes([raw[0] ^ 1]) + raw[1:])
    elif mutation == "hidden-file":
        (built_attempt.final_evidence / ".foreign").write_bytes(b"foreign")
    else:
        (built_attempt.final_evidence / "unexpected-empty-directory").mkdir()
    before = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="tampered partial evidence root",
    )

    with pytest.raises((ValueError, json.JSONDecodeError)):
        built_attempt.capture()
    after = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="tampered partial evidence root",
    )
    assert after == before
    _assert_no_consumable_aggregate(built_attempt)


@pytest.mark.parametrize(
    "occupant",
    ["empty-directory", "foreign-directory", "regular-file"],
)
def test_concurrent_evidence_root_occupant_is_adopted_only_if_empty(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    occupant: str,
) -> None:
    _crash_after_publication_owner(built_attempt, monkeypatch)
    original = control._ensure_evidence_directory
    injected = False
    occupied_raw = b"foreign-regular-file"

    def race_at_root(path: Path, *, name: str) -> bool:
        nonlocal injected
        if path.absolute() == built_attempt.final_evidence.absolute() and not injected:
            injected = True
            if occupant == "regular-file":
                path.write_bytes(occupied_raw)
            else:
                path.mkdir()
                if occupant == "foreign-directory":
                    (path / ".foreign").write_bytes(b"foreign")
        return original(path, name=name)

    monkeypatch.setattr(control, "_ensure_evidence_directory", race_at_root)
    if occupant == "empty-directory":
        built_attempt.capture()
        built_attempt.verify()
        return

    with pytest.raises(ValueError):
        built_attempt.capture()
    if occupant == "regular-file":
        assert built_attempt.final_evidence.read_bytes() == occupied_raw
    else:
        assert (built_attempt.final_evidence / ".foreign").read_bytes() == b"foreign"
    _assert_no_consumable_aggregate(built_attempt)


@pytest.mark.parametrize("competitor_bytes", ["exact", "wrong"])
def test_concurrent_evidence_file_occupant_is_adopted_only_if_exact(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    competitor_bytes: str,
) -> None:
    original = control.os.link
    target_relative = "aggregation-attempt/READY"
    injected = False

    def race_at_link(
        source: Path,
        destination: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal injected
        destination_path = Path(destination)
        if _evidence_relative(built_attempt, destination_path) == target_relative and not injected:
            injected = True
            raw = Path(source).read_bytes()
            if competitor_bytes == "wrong":
                raw = bytes([raw[0] ^ 1]) + raw[1:]
            destination_path.write_bytes(raw)
        original(source, destination, *args, **kwargs)

    monkeypatch.setattr(control.os, "link", race_at_link)
    if competitor_bytes == "exact":
        built_attempt.capture()
        built_attempt.verify()
        return

    with pytest.raises(ValueError, match="differs"):
        built_attempt.capture()
    wrong = built_attempt.final_evidence / "aggregation-attempt" / "READY"
    assert wrong.is_file()
    _assert_no_consumable_aggregate(built_attempt)


@pytest.mark.parametrize("link_phase", ["before", "after"])
def test_evidence_file_link_crash_keeps_staging_outside_authoritative_tree_and_resumes(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    link_phase: str,
) -> None:
    original = control.os.link
    target_relative = "configs/common_beta_post_recovery_calibration.yaml"
    fired = False

    def crash_at_link(
        source: Path,
        destination: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal fired
        if _evidence_relative(built_attempt, Path(destination)) == target_relative and not fired:
            fired = True
            assert _evidence_relative(built_attempt, Path(source)) is None
            if link_phase == "after":
                original(source, destination, *args, **kwargs)
            raise InjectedCrash(f"{link_phase}-evidence-link")
        original(source, destination, *args, **kwargs)

    monkeypatch.setattr(control.os, "link", crash_at_link)
    with pytest.raises(InjectedCrash, match="evidence-link"):
        built_attempt.capture()
    target = built_attempt.final_evidence.joinpath(*target_relative.split("/"))
    assert target.exists() is (link_phase == "after")
    assert list(
        built_attempt.aggregate.parent.glob(f".{built_attempt.final_evidence.name}.publishing-*")
    )
    assert not any(".staged-" in child.name for child in built_attempt.final_evidence.rglob("*"))
    _assert_no_consumable_aggregate(built_attempt)

    built_attempt.capture()
    built_attempt.verify()


@pytest.mark.parametrize("fsync_phase", ["tree", "parent"])
def test_complete_evidence_fsync_crash_resumes_exactly(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    fsync_phase: str,
) -> None:
    fired = False
    if fsync_phase == "tree":
        original_tree = control._fsync_tree

        def fsync_tree_then_crash(path: Path) -> None:
            nonlocal fired
            original_tree(path)
            if path.absolute() == built_attempt.final_evidence.absolute() and not fired:
                fired = True
                raise InjectedCrash("after-complete-evidence-tree-fsync")

        monkeypatch.setattr(control, "_fsync_tree", fsync_tree_then_crash)
    else:
        original_directory = control._fsync_directory

        def fsync_parent_then_crash(path: Path) -> None:
            nonlocal fired
            original_directory(path)
            if (
                path.absolute() == built_attempt.final_evidence.parent.absolute()
                and built_attempt.final_evidence.is_dir()
                and len(
                    control._directory_tree_manifest(
                        built_attempt.final_evidence,
                        name="complete evidence tree",
                    )["files"]
                )
                == 19
                and not fired
            ):
                fired = True
                raise InjectedCrash("after-complete-evidence-parent-fsync")

        monkeypatch.setattr(control, "_fsync_directory", fsync_parent_then_crash)

    with pytest.raises(InjectedCrash, match="complete-evidence"):
        built_attempt.capture()
    _assert_no_consumable_aggregate(built_attempt)
    built_attempt.capture()
    built_attempt.verify()


def test_post_fsync_concurrent_extra_is_caught_before_consumable_publication(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = control._fsync_tree
    injected = False

    def fsync_then_inject(path: Path) -> None:
        nonlocal injected
        original(path)
        if path.absolute() == built_attempt.final_evidence.absolute() and not injected:
            injected = True
            (path / ".after-fsync-race").write_bytes(b"foreign")

    monkeypatch.setattr(control, "_fsync_tree", fsync_then_inject)
    with pytest.raises(ValueError, match="failed final verification"):
        built_attempt.capture()
    assert (built_attempt.final_evidence / ".after-fsync-race").is_file()
    _assert_no_consumable_aggregate(built_attempt)
    with pytest.raises(ValueError):
        built_attempt.capture()
    _assert_no_consumable_aggregate(built_attempt)


def test_evidence_claim_binds_payload_without_self_reference_and_publication_binds_full_tree(
    built_attempt: BuiltAttempt,
) -> None:
    built_attempt.capture()
    claim_path = built_attempt.final_evidence.joinpath(*_evidence_claim_relative().split("/"))
    claim = json.loads(claim_path.read_bytes())
    full_tree = control._directory_tree_manifest(
        built_attempt.final_evidence,
        name="published aggregate evidence",
    )
    payload_tree = control._evidence_tree_without_claim(full_tree)
    owner = control.parse_post_recovery_aggregate_publication_owner(
        Path(f"{built_attempt.aggregate}.ATTEMPT")
    )
    publication = control.parse_post_recovery_aggregate_publication_receipt(
        Path(f"{built_attempt.aggregate}.PUBLISHED")
    )

    assert claim_path.read_bytes() == control._canonical_json(claim)
    assert claim["created_at_utc"] == owner["created_at_utc"]
    assert claim["publication_owner_receipt_sha256"] == _sha(
        Path(f"{built_attempt.aggregate}.ATTEMPT")
    )
    assert claim["payload_exact_tree_manifest_sha256"] == (
        control._evidence_tree_manifest_sha256(payload_tree)
    )
    assert publication["aggregate_evidence_manifest_sha256"] == (
        control._evidence_tree_manifest_sha256(full_tree)
    )
    built_attempt.verify()


@pytest.mark.parametrize(
    "relative",
    [
        "configs/common_beta_post_recovery_calibration.yaml",
        "configs/common_beta_pilot_base.yaml",
        "submission-registry/intent.json",
        "submission-registry/submission.json",
        "aggregate-submission/intent.json",
        "aggregate-submission/attempt.json",
        "aggregate-submission/failure-chain.json",
        "aggregate-submission/script.sbatch",
        "aggregate-submission/controller/attempt-0001.sbatch",
        "aggregate-submission/controller/attempt-0002.sbatch",
        "aggregate-submission/attempts/attempt-0001.json",
        "aggregate-submission/attempts/attempt-0002.json",
        "aggregate-submission/failures/job-2999.json",
        "aggregate-submission/failures/job-2999.sacct.psv",
        "aggregation-attempt/READY",
        "aggregation-attempt/EVIDENCE_CLAIM.json",
        "aggregation-attempt/AUTHORITY.json",
        "aggregation-attempt/AUTHORITY.squeue.txt",
        "aggregation-attempt/AUTHORITY.sacct.psv",
    ],
)
def test_completed_bundle_tampering_fails_closed(
    built_attempt: BuiltAttempt,
    relative: str,
) -> None:
    built_attempt.capture()
    target = built_attempt.final_evidence.joinpath(*relative.split("/"))
    _force_write(target, target.read_bytes() + b"x")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        built_attempt.verify()


def test_completed_bundle_rejects_even_hidden_extra_file(
    built_attempt: BuiltAttempt,
) -> None:
    built_attempt.capture()
    hidden = built_attempt.final_evidence / ".unbound"
    hidden.write_bytes(b"extra")
    with pytest.raises(ValueError, match="unexpected"):
        built_attempt.verify()


@pytest.mark.parametrize(
    "authority_mutation",
    [
        "extra_completed",
        "prior_failure_became_completed",
        "extra_live",
        "missing_selected",
    ],
)
def test_ambiguous_scheduler_authority_fails_before_claim(
    built_attempt: BuiltAttempt,
    authority_mutation: str,
) -> None:
    if authority_mutation == "extra_completed":
        built_attempt.scheduler.rows["3001"] = _cpu_history_row(
            job_id="3001",
            job_name=built_attempt.scheduler.job_name,
            state="COMPLETED",
        )
    elif authority_mutation == "prior_failure_became_completed":
        built_attempt.scheduler.rows["2999"] = _cpu_history_row(
            job_id="2999",
            job_name=built_attempt.scheduler.job_name,
            state="COMPLETED",
        )
    elif authority_mutation == "extra_live":
        built_attempt.scheduler.live_ids = ["3001"]
    elif authority_mutation == "missing_selected":
        del built_attempt.scheduler.rows["3000"]
    else:  # pragma: no cover
        raise AssertionError(authority_mutation)

    with pytest.raises(ValueError):
        built_attempt.capture()
    assert not Path(f"{built_attempt.aggregate}.ATTEMPT").exists()
    assert not built_attempt.aggregate.exists()
    assert not built_attempt.final_evidence.exists()


def test_unregistered_ready_attempt_cannot_claim_final_namespace(
    built_attempt: BuiltAttempt,
) -> None:
    with pytest.raises(ValueError, match="missing"):
        control.capture_post_recovery_aggregate_terminal_evidence(
            built_attempt.aggregate,
            attempt_job_id="3001",
        )
    assert not Path(f"{built_attempt.aggregate}.ATTEMPT").exists()


def test_raw_only_replays_after_same_fresh_terminal_query(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _post_file_install_crash(monkeypatch, name="raw aggregation sacct evidence")
    with pytest.raises(InjectedCrash):
        built_attempt.capture()
    raw_path = Path(f"{built_attempt.aggregate}.TERMINAL.sacct.psv")
    assert raw_path.is_file()
    assert not Path(f"{built_attempt.aggregate}.TERMINAL.json").exists()

    built_attempt.capture()
    built_attempt.verify()


@pytest.mark.parametrize("mode", ["query_error", "query_mismatch", "disk_mismatch"])
def test_raw_only_requires_exact_fresh_reconfirmation(
    built_attempt: BuiltAttempt,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _post_file_install_crash(monkeypatch, name="raw aggregation sacct evidence")
    with pytest.raises(InjectedCrash):
        built_attempt.capture()
    raw_path = Path(f"{built_attempt.aggregate}.TERMINAL.sacct.psv")
    if mode == "query_error":
        built_attempt.scheduler.terminal_error = True
        expected = RuntimeError
    elif mode == "query_mismatch":
        built_attempt.scheduler.terminal_raw = built_attempt.scheduler.terminal_raw.replace(
            b"|COMPLETED|0:0|0:0|",
            b"|FAILED|1:0|1:0|",
        )
        expected = ValueError
    else:
        raw_path.write_bytes(raw_path.read_bytes() + b"x")
        expected = ValueError

    with pytest.raises(expected):
        built_attempt.capture()
    assert not Path(f"{built_attempt.aggregate}.TERMINAL.json").exists()
    assert not Path(f"{built_attempt.aggregate}.SUCCESS").exists()


@pytest.mark.parametrize(
    "occupied_suffix",
    [
        "",
        ".ATTEMPT",
        ".PUBLISHED",
        ".TERMINAL.sacct.psv",
        ".TERMINAL.json",
        ".SUCCESS",
    ],
)
def test_precreated_truncated_final_artifact_is_never_overwritten(
    built_attempt: BuiltAttempt,
    occupied_suffix: str,
) -> None:
    occupied = (
        built_attempt.aggregate
        if not occupied_suffix
        else Path(f"{built_attempt.aggregate}{occupied_suffix}")
    )
    occupied.write_bytes(b"truncated")
    before = occupied.read_bytes()
    with pytest.raises(ValueError):
        built_attempt.capture()
    assert occupied.read_bytes() == before
    if occupied_suffix != ".SUCCESS":
        assert not Path(f"{built_attempt.aggregate}.SUCCESS").exists()


def test_success_is_self_contained_after_live_registry_and_staging_archive(
    built_attempt: BuiltAttempt,
    tmp_path: Path,
) -> None:
    completed = built_attempt.capture()
    archive = tmp_path / "archived-attempt-control"
    built_attempt.campaign_root.rename(archive)
    built_attempt.scheduler.fail_all = True

    verified = built_attempt.verify()
    replayed = built_attempt.capture()
    assert verified["receipt_sha256"] == completed["receipt_sha256"]
    assert replayed["receipt_sha256"] == completed["receipt_sha256"]
    assert archive.is_dir()
