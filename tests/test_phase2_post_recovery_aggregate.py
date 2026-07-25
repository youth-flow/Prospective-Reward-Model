from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import smart_reward.phase2_config as phase2_config_module
import smart_reward.phase2_pilot_aggregate as pilot_reader
import smart_reward.phase2_post_recovery_aggregate as aggregate
import smart_reward.phase2_post_recovery_control as control
import smart_reward.phase2_post_recovery_output as output_verifier
from smart_reward.phase2_aggregate import validate_post_recovery_pilot_head_training
from smart_reward.phase2_config import phase2_design_identity, validate_phase2_config
from smart_reward.phase2_post_recovery_control import (
    OPTIMIZER_SCHEDULE_SHA256,
    POST_RECOVERY_OUTPUT_VERIFICATION_SCHEMA,
    POST_RECOVERY_RUN_STATUS_SCHEMA,
)
from smart_reward.phase2_rollout import _sanitize_pilot_training_audit
from smart_reward.phase2_training import compile_phase2_training_settings

ROOT = Path(__file__).resolve().parents[1]
PILOT_SUBMISSION_INTENT_SHA256 = "6" * 64
PILOT_SUBMISSION_LEDGER_SHA256 = "7" * 64


def _pilot_helpers():
    path = ROOT / "tests" / "test_phase2_pilot_aggregate.py"
    spec = importlib.util.spec_from_file_location("_pilot_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aggregate_helpers():
    path = ROOT / "tests" / "test_phase2_aggregate.py"
    spec = importlib.util.spec_from_file_location("_aggregate_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recovery_helpers():
    path = ROOT / "tests" / "test_phase2_recovery_aggregate.py"
    spec = importlib.util.spec_from_file_location("_recovery_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_helpers():
    path = ROOT / "tests" / "test_phase2_config.py"
    spec = importlib.util.spec_from_file_location("_config_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _link_directory(link: Path, target: Path) -> None:
    if os.name != "nt":
        link.symlink_to(target, target_is_directory=True)
        return
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise OSError(f"could not create test directory junction: {completed.stderr}")


def _script_helper(filename: str, module_name: str):
    path = ROOT / "scripts" / "hpc4" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _rewrite_run_submission_hashes(
    results: list[Path],
    *,
    intent_sha256: str,
    submission_sha256: str,
) -> None:
    for result in results:
        marker_path = result.parent / "SUCCESS"
        marker: dict[str, str] = {}
        for line in marker_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split("=", 1)
            marker[key] = value
        marker["submission_intent_sha256"] = intent_sha256
        marker["submission_ledger_sha256"] = submission_sha256
        marker_path.write_bytes(control._aggregate_receipt_bytes(marker))


def _materialize_pilot_submission_evidence(
    *,
    project: Path,
    evidence_root: Path,
    config: dict[str, object],
    overlay_relative: str,
    overlay_sha256: str,
    base_relative: str,
    base_sha256: str,
    authorization_sha256: str,
    array_job_id: str,
) -> tuple[str, str]:
    helper = _script_helper(
        "submit_phase2_post_recovery_array_once.py",
        f"_post_recovery_array_helper_{array_job_id}_{evidence_root.name}",
    )
    pilot_phase = str(config["design"]["pilot_phase"])
    design_sha256 = phase2_design_identity(config)
    producer_commit = "a" * 40
    image_sha256 = "b" * 64
    inventory_sha256 = "c" * 64
    repository_root = ROOT.absolute()
    hf_cache = "/project/test-hf-cache"
    export_spec = ",".join(
        (
            "PATH=/usr/local/bin:/usr/bin:/bin",
            f"PRORM_PROJECT_ROOT={project}",
            f"PRORM_REPO_ROOT={repository_root}",
            "PRORM_SCRATCH_ROOT=/scratch/test-prorm",
            "PRORM_IMAGE=/project/test-prorm.sif",
            f"PRORM_IMAGE_SHA256={image_sha256}",
            f"PRORM_HF_CACHE={hf_cache}",
            (
                "PRORM_HF_INVENTORY="
                f"{hf_cache}/inventories/{config['design']['source_config_hash']}.json"
            ),
            f"PRORM_HF_INVENTORY_SHA256={inventory_sha256}",
            f"PRORM_POST_RECOVERY_OVERLAY_REL={overlay_relative}",
            f"PRORM_PHASE2_BASE_REL={base_relative}",
            f"PRORM_POST_RECOVERY_OVERLAY_SHA256={overlay_sha256}",
            f"PRORM_PHASE2_BASE_SHA256={base_sha256}",
            f"PRORM_POST_RECOVERY_DESIGN_SHA256={design_sha256}",
            (f"PRORM_PHASE2_BASE_CONFIG_HASH={config['design']['source_config_hash']}"),
            "PRORM_RECOVERY_AUTHORIZATION=/project/recovery-authorization.json",
            f"PRORM_RECOVERY_AUTHORIZATION_SHA256={authorization_sha256}",
            f"PRORM_OPTIMIZER_SCHEDULE_SHA256={OPTIMIZER_SCHEDULE_SHA256}",
            f"PRORM_GIT_COMMIT={producer_commit}",
            f"PRORM_POST_RECOVERY_PILOT_PHASE={pilot_phase}",
            f"PRORM_POST_RECOVERY_NAMESPACE={pilot_phase}",
            "PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=0",
            "PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=0",
        )
    )
    job_name = f"prorm-p2-post-{pilot_phase}-{design_sha256[:12]}"
    script_relative = "scripts/hpc4/phase2_post_recovery_calibration.sbatch"
    script = repository_root.joinpath(*Path(script_relative).parts)
    intent = helper._intent_payload(
        pilot_phase=pilot_phase,
        design_sha256=design_sha256,
        base_config_hash=str(config["design"]["source_config_hash"]),
        authorization_sha256=authorization_sha256,
        optimizer_schedule_sha256=OPTIMIZER_SCHEDULE_SHA256,
        git_commit=producer_commit,
        image_sha256=image_sha256,
        inventory_sha256=inventory_sha256,
        overlay_sha256=overlay_sha256,
        base_file_sha256=base_sha256,
        sbatch_script_relative=script_relative,
        sbatch_script_sha256=_sha(script),
        export_spec=export_spec,
        export_spec_sha256=hashlib.sha256(export_spec.encode()).hexdigest(),
        walltime="12:00:00",
        job_name=job_name,
        project_root=os.fspath(project),
        repository_root=os.fspath(repository_root),
        submitter_user="tester",
        created_at_utc="2026-07-25T00:00:00Z",
    )
    intent_raw = _canonical_json_bytes(intent)
    intent_sha256 = hashlib.sha256(intent_raw).hexdigest()
    command = repository_root / script_relative
    raw_scontrol = (
        f"JobId={array_job_id} ArrayJobId={array_job_id} ArrayTaskId=0-2%2 "
        f"ArrayTaskThrottle=2 JobName={job_name} UserId=tester(1) "
        "Account=sigroup QOS=l20_qos JobState=PENDING Reason=JobHeldUser "
        "Requeue=0 Restarts=0 Partition=gpu-l20 NumNodes=1 NumTasks=1 "
        "NumCPUs=8 CPUs/Task=8 MinMemoryNode=96G TimeLimit=12:00:00 "
        f"Command={command} WorkDir={repository_root} "
        "TRES=cpu=8,mem=96G,node=1,gres/gpu=1 TresPerNode=gres:gpu:1\n"
    )
    state, scheduler = helper._parse_scontrol_records(
        raw_scontrol,
        array_job_id=array_job_id,
        expected_name=job_name,
        expected_walltime="12:00:00",
        expected_command=command,
        expected_workdir=repository_root,
        expected_user="tester",
    )
    assert state == "HELD" and scheduler is not None
    submission = helper._submission_payload(
        intent=intent,
        intent_sha256=intent_sha256,
        array_job_id=array_job_id,
        submitted_cluster="hpc4",
        scheduler_request=scheduler,
    )
    submission_raw = _canonical_json_bytes(submission)
    submission_sha256 = hashlib.sha256(submission_raw).hexdigest()
    registry = evidence_root / "submission-registry"
    registry.mkdir(parents=True)
    (registry / "intent.json").write_bytes(intent_raw)
    (registry / "submission.json").write_bytes(submission_raw)
    return intent_sha256, submission_sha256


def _terminalize_staged_aggregate(
    *,
    monkeypatch: pytest.MonkeyPatch,
    project: Path,
    aggregate_path: Path,
    attempt_root: Path,
    config: dict[str, object],
    array_job_id: str,
    pilot_intent_sha256: str,
    pilot_submission_sha256: str,
) -> None:
    helper = _script_helper(
        "submit_phase2_post_recovery_aggregate_attempt.py",
        f"_post_recovery_cpu_helper_{array_job_id}_{aggregate_path.name}",
    )
    repository_root = ROOT.absolute()
    design_sha256 = phase2_design_identity(config)
    pilot_phase = str(config["design"]["pilot_phase"])
    cpu_job_id = "1"
    registry = (
        project
        / "runs"
        / "phase2-post-recovery-aggregate-attempts"
        / aggregate_path.name
        / "submission-registry"
    )
    (registry / "attempts").mkdir(parents=True)
    (registry / "failures").mkdir()
    workload_export = ",".join(
        (
            f"PRORM_PROJECT_ROOT={project}",
            f"PRORM_REPO_ROOT={repository_root}",
            f"PRORM_POST_RECOVERY_DESIGN_SHA256={design_sha256}",
            f"PRORM_POST_RECOVERY_ARRAY_JOB_ID={array_job_id}",
            f"PRORM_AGGREGATOR_GIT_COMMIT={'f' * 40}",
            f"PRORM_POST_RECOVERY_AGGREGATE_OUTPUT={aggregate_path}",
            f"PRORM_POST_RECOVERY_PILOT_PHASE={pilot_phase}",
        )
    )
    job_name = f"prorm-p2-post-agg-{design_sha256[:12]}-{array_job_id}"
    script_relative = "scripts/hpc4/phase2_post_recovery_aggregate.sbatch"
    script = repository_root / script_relative
    intent = helper._intent_payload(
        pilot_phase=pilot_phase,
        design_sha256=design_sha256,
        pilot_array_job_id=array_job_id,
        aggregator_git_commit="f" * 40,
        project_root=project,
        repository_root=repository_root,
        output=aggregate_path,
        partition="amd",
        walltime="01:00:00",
        workload_export_spec=workload_export,
        script_relative=script_relative,
        script_sha256=_sha(script),
        submitter_user="tester",
        job_name=job_name,
        created_at_utc="2026-07-25T00:00:00Z",
    )
    intent_raw = _canonical_json_bytes(intent)
    intent_sha256 = hashlib.sha256(intent_raw).hexdigest()
    control_suffix = (
        f",PRORM_POST_RECOVERY_AGGREGATE_SUBMISSION_REGISTRY={registry}"
        f",PRORM_POST_RECOVERY_AGGREGATE_INTENT_SHA256={intent_sha256}"
        ",PRORM_POST_RECOVERY_AGGREGATE_ATTEMPT_INDEX=1"
        ",PRORM_POST_RECOVERY_AGGREGATE_WORKLOAD_EXPORT_SHA256="
        f"{intent['workload_export_spec_sha256']}"
    )
    scheduler_export = f"{workload_export}{control_suffix}"
    raw_scontrol = (
        f"JobId={cpu_job_id} JobName={job_name} UserId=tester(1) "
        "Account=sigroup JobState=PENDING Reason=JobHeldUser Requeue=0 "
        "Restarts=0 Partition=amd NumNodes=1 NumTasks=1 NumCPUs=4 "
        "CPUs/Task=4 MinMemoryNode=16G TimeLimit=01:00:00 "
        f"Command={script} WorkDir={repository_root} "
        f"Comment=prorm-aggregate:{intent_sha256}:attempt-1 "
        "TRES=cpu=4,mem=16G,node=1 TresPerNode=\n"
    )
    state, scheduler = helper._parse_scontrol(
        raw_scontrol,
        job_id=cpu_job_id,
        intent=intent,
        intent_sha256=intent_sha256,
        attempt_index=1,
        script=script,
        repository_root=repository_root,
    )
    assert state == "HELD" and scheduler is not None
    attempt = helper._attempt_payload(
        intent=intent,
        intent_sha256=intent_sha256,
        attempt_index=1,
        job_id=cpu_job_id,
        scheduler_export_spec=scheduler_export,
        scheduler_request=scheduler,
    )
    attempt_raw = _canonical_json_bytes(attempt)
    attempt_sha256 = hashlib.sha256(attempt_raw).hexdigest()
    (registry / "intent.json").write_bytes(intent_raw)
    (registry / "attempts" / "attempt-0001.json").write_bytes(attempt_raw)

    evidence = attempt_root / "evidence"
    cpu_evidence = evidence / "aggregate-submission"
    (cpu_evidence / "attempts").mkdir(parents=True)
    (cpu_evidence / "failures").mkdir()
    (cpu_evidence / "intent.json").write_bytes(intent_raw)
    (cpu_evidence / "attempt.json").write_bytes(attempt_raw)
    (cpu_evidence / "attempts" / "attempt-0001.json").write_bytes(attempt_raw)
    failure_chain_raw = _canonical_json_bytes([])
    (cpu_evidence / "failure-chain.json").write_bytes(failure_chain_raw)
    staged_aggregate = attempt_root / "aggregate.json"
    overlay_name = aggregate._semantic_lineage_filenames(
        json.loads(staged_aggregate.read_text(encoding="utf-8"))
    )[0]
    overlay = evidence / "configs" / overlay_name
    base = evidence / "configs" / "common_beta_pilot_base.yaml"
    ready = {
        "schema_version": control.POST_RECOVERY_AGGREGATE_ATTEMPT_READY_SCHEMA,
        "status": "READY",
        "slurm_job_id": cpu_job_id,
        "slurm_job_is_array": "false",
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "amd",
        "restart_count": "0",
        "pilot_array_job_id": array_job_id,
        "pilot_phase": pilot_phase,
        "phase2_design_sha256": design_sha256,
        "base_config_hash": str(config["design"]["source_config_hash"]),
        "recovery_authorization_sha256": str(
            config["recovery_success_reference"]["artifact_sha256"]
        ),
        "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
        "pilot_terminal_evidence_sha256": str(
            json.loads(staged_aggregate.read_text(encoding="utf-8"))["post_recovery_control"][
                "pilot_terminal_evidence_sha256"
            ]
        ),
        "submission_intent_sha256": pilot_intent_sha256,
        "submission_ledger_sha256": pilot_submission_sha256,
        "aggregate_submission_intent_sha256": intent_sha256,
        "aggregate_submission_attempt_sha256": attempt_sha256,
        "aggregate_submission_attempt_index": "1",
        "aggregate_submission_failure_chain_sha256": hashlib.sha256(failure_chain_raw).hexdigest(),
        "phase2_overlay_sha256": _sha(overlay),
        "phase2_base_sha256": _sha(base),
        "aggregator_git_commit": "f" * 40,
        "producer_git_commit": "a" * 40,
        "image_sha256": "b" * 64,
        "hf_inventory_sha256": "c" * 64,
        "final_output": os.fspath(aggregate_path),
        "final_evidence_root": f"{aggregate_path}.evidence",
        "attempt_aggregate": "aggregate.json",
        "attempt_evidence": "evidence",
        "aggregate_sha256": _sha(staged_aggregate),
        "final_namespace_untouched": "true",
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    (attempt_root / "READY").write_bytes(control._aggregate_receipt_bytes(ready))

    authority_raw = (
        f"{cpu_job_id}|{cpu_job_id}|{job_name}|COMPLETED|0:0|0:0|"
        "hpc4|sigroup|amd|1|4|2026-07-25T00:01:00|01:00:00|"
        "billing=4,cpu=4,mem=16G,node=1|billing=4,cpu=4,mem=16G,node=1\n"
    )
    terminal_raw = (
        f"{cpu_job_id}|{cpu_job_id}|COMPLETED|0:0|0:0|hpc4|sigroup|amd|"
        "1|4|billing=4,cpu=4,mem=16G,node=1|"
        "billing=4,cpu=4,mem=16G,node=1\n"
    ).encode()
    previous_run = subprocess.run

    def scheduler_run(
        command: list[str] | tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess:
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "sacct" and any(str(item).startswith("--name=") for item in command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=authority_raw,
                stderr="",
            )
        assert command[0] == "sacct" and "-j" in command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=terminal_raw,
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", scheduler_run)
    try:
        control.capture_post_recovery_aggregate_terminal_evidence(
            aggregate_path,
            attempt_job_id=cpu_job_id,
        )
    finally:
        monkeypatch.setattr(subprocess, "run", previous_run)


def _rebind_final_aggregate_receipts(aggregate_path: Path) -> None:
    aggregate_sha256 = _sha(aggregate_path)
    ready_path = Path(f"{aggregate_path}.evidence") / "aggregation-attempt" / "READY"
    ready = control.parse_post_recovery_aggregate_attempt_ready(ready_path)
    ready["aggregate_sha256"] = aggregate_sha256
    ready_path.write_bytes(control._aggregate_receipt_bytes(ready))
    ready_sha256 = _sha(ready_path)

    authority_path = ready_path.parent / "AUTHORITY.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["attempt_ready_sha256"] = ready_sha256
    _write_canonical_json(authority_path, authority)
    authority_sha256 = _sha(authority_path)
    evidence_tree = control._directory_tree_manifest(
        Path(f"{aggregate_path}.evidence"),
        name="test aggregate evidence",
    )
    evidence_tree_sha256 = control._evidence_tree_manifest_sha256(evidence_tree)

    owner_path = Path(f"{aggregate_path}.ATTEMPT")
    owner = control.parse_post_recovery_aggregate_publication_owner(owner_path)
    owner["attempt_ready_sha256"] = ready_sha256
    owner["aggregate_sha256"] = aggregate_sha256
    owner_path.write_bytes(control._aggregate_receipt_bytes(owner))

    publication_path = Path(f"{aggregate_path}.PUBLISHED")
    publication = control.parse_post_recovery_aggregate_publication_receipt(publication_path)
    publication["aggregate_sha256"] = aggregate_sha256
    publication["aggregate_attempt_ready_sha256"] = ready_sha256
    publication["aggregate_submission_authority_sha256"] = authority_sha256
    publication["aggregate_evidence_manifest_sha256"] = evidence_tree_sha256
    publication_path.write_bytes(control._aggregate_receipt_bytes(publication))
    publication_sha256 = _sha(publication_path)

    terminal_path = Path(f"{aggregate_path}.TERMINAL.json")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["aggregate"]["sha256"] = aggregate_sha256
    terminal["aggregate"]["size_bytes"] = aggregate_path.stat().st_size
    terminal["publication_receipt"]["sha256"] = publication_sha256
    terminal["publication_receipt"]["size_bytes"] = publication_path.stat().st_size
    _write_canonical_json(terminal_path, terminal)
    terminal_sha256 = _sha(terminal_path)

    success_path = Path(f"{aggregate_path}.SUCCESS")
    success = dict(publication)
    success.update(
        {
            "schema_version": control.POST_RECOVERY_AGGREGATE_SUCCESS_SCHEMA,
            "status": "SUCCESS",
            "aggregate_publication_receipt_sha256": publication_sha256,
            "aggregation_terminal_evidence_sha256": terminal_sha256,
            "created_at_utc": "2026-07-25T00:02:00Z",
        }
    )
    success_path.write_bytes(control._aggregate_receipt_bytes(success))


def _redacted_head_training(
    *,
    helpers: object,
    config: dict[str, object],
    seed: int,
    design_sha256: str,
    train_oracle_reward_sha256: str,
) -> dict[str, object]:
    head_weights = {
        "bt_mle": helpers._fixture_head(1.0, 0.0),
        "prorm_plus": helpers._fixture_head(0.0, 1.0),
    }
    audit = helpers._head_training_audit(
        seed=seed,
        config=config,
        design_sha=design_sha256,
        train_oracle_reward_sha=train_oracle_reward_sha256,
        head_weights=head_weights,
    )
    settings_sha256 = compile_phase2_training_settings(config).sha256
    audit["training_settings_sha256"] = settings_sha256
    primary_audit = audit["primary_optimization_audit"]
    low = audit["low_dimensional_control"]
    training_instance = {
        "schema_version": "phase2-training-instance/v1",
        "phase2_config_hash": design_sha256,
        "settings_sha256": settings_sha256,
        "input_training_sha256": audit["input_training_sha256"],
        "oracle_reward_sha256": train_oracle_reward_sha256,
        "seed": seed,
        "label_stream_sha256": audit["label_stream"]["label_stream_sha256"],
        "reward_head_identifiability_sha256": helpers._canonical_sha256(
            primary_audit["reward_head_identifiability"]
        ),
        "prorm_moment_map_identifiability_sha256": helpers._canonical_sha256(
            primary_audit["prorm_moment_map_identifiability"]
        ),
        "bt_head_sha256": audit["primary_heads"]["bt_mle"]["head_sha256"],
        "prorm_plus_head_sha256": audit["primary_heads"]["prorm_plus"]["head_sha256"],
        "low_dimensional_head_sha256": low["head"]["head_sha256"],
        "low_dimensional_projection_sha256": low["projection"]["projection_sha256"],
        "low_dimensional_moment_map_identifiability_sha256": (
            helpers._canonical_sha256(low["projected_prorm_moment_map_identifiability"])
        ),
        "exact_margin_head_sha256": audit["exact_margin_control"]["head"]["head_sha256"],
        "exact_soft_label_bt_head_sha256": audit["exact_soft_label_bt_control"]["head"][
            "head_sha256"
        ],
        "direct_oracle_direction_sha256": audit["direct_oracle_identity"][
            "native_oracle_direction"
        ]["direction_sha256"],
    }
    audit["training_instance_sha256"] = helpers._canonical_sha256(training_instance)
    return {
        "training_arm": "r4_independent_gamma_0.9",
        "training_design_sha256": design_sha256,
        "heads_sha256": helpers._canonical_sha256(head_weights),
        "audit": _sanitize_pilot_training_audit(audit),
        "source": "trained_after_train_oracle_rescore",
        "old_phase1_comparison_heads_reused": False,
        "test_data_accessed": False,
        "head_weights_serialized": False,
        "audit_vector_fields_redacted": sorted(
            {
                "head_weight",
                "head_weights",
                "direction",
                "natural_direction",
                "displacement",
                "oracle_displacement",
                "moment",
                "operator_direction",
                "projection_matrix",
                "true_rewards",
            }
        ),
    }


def _upgrade_run(
    run: Path,
    *,
    helpers: object,
    config: dict[str, object],
    seed: int,
    task: int,
    authorization_sha256: str,
    array_job_id: str = "1000",
    raw_job_id_base: int = 7000,
) -> None:
    pilot_phase = str(config["design"]["pilot_phase"])
    result = run / "phase2-pilot-diagnostics.json"
    old_output = run / "phase2-output-verification.json"
    strict_output = run / "post-recovery-output-verification.json"
    result_value = json.loads(result.read_text(encoding="utf-8"))
    design_sha256 = result_value["phase2_design_sha256"]
    train_oracle_reward_sha256 = helpers._token_sha256(f"{seed}:train-oracle-rewards")
    result_value["train_oracle_rescore"] = {
        "source": "saved_train_candidates_rescored_with_pinned_oracle",
        "num_prompts": config["run"]["split_sizes"]["train"],
        "num_candidates": config["data"]["num_candidates"],
        "transformed_rewards_sha256": train_oracle_reward_sha256,
        "oracle_chat_template_sha256": helpers._token_sha256("oracle-chat-template"),
        "frozen_transform": {"b": 0.0, "tau": 1.0},
        "raw_oracle_logits_serialized": False,
    }
    result_value["head_training"] = _redacted_head_training(
        helpers=helpers,
        config=config,
        seed=seed,
        design_sha256=design_sha256,
        train_oracle_reward_sha256=train_oracle_reward_sha256,
    )
    _write_json(result, result_value)
    training_gate = validate_post_recovery_pilot_head_training(
        result_value["head_training"],
        config=config,
        design_sha256=design_sha256,
        seed=seed,
        train_oracle_reward_sha256=train_oracle_reward_sha256,
    )
    training = training_gate["five_head_training"]
    _write_json(
        strict_output,
        {
            "schema_version": POST_RECOVERY_OUTPUT_VERIFICATION_SCHEMA,
            "status": "passed",
            "pilot_phase": pilot_phase,
            "slurm_job_id_raw": str(raw_job_id_base + task),
            "allocation_job_id_raw": str(raw_job_id_base + task),
            "slurm_array_task_job_id": f"{array_job_id}_{task}",
            "array_job_id": array_job_id,
            "array_task_id": str(task),
            "seed": seed,
            "phase2_design_sha256": design_sha256,
            "source_config_hash": result_value["source_config_hash"],
            "result_sha256": _sha(result),
            "phase2_output_verification_sha256": _sha(old_output),
            "diagnostics_sha256": _sha(run / "phase2-pilot-diagnostics.diagnostics.jsonl"),
            "recovery_authorization_sha256": authorization_sha256,
            "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
            "materialization_mode": "fresh",
            "recovery_outputs_reused": False,
            "five_head_adopted_schedule_verified": True,
            "five_head_training": training,
            "target_free_information_boundary_verified": True,
        },
    )
    marker = {
        "schema_version": POST_RECOVERY_RUN_STATUS_SCHEMA,
        "status": "SUCCESS",
        "pilot_phase": pilot_phase,
        "workload_exit_code": "0",
        "final_exit_code": "0",
        "slurm_job_id": str(raw_job_id_base + task),
        "allocation_job_id_raw": str(raw_job_id_base + task),
        "slurm_array_task_job_id": f"{array_job_id}_{task}",
        "array_job_id": array_job_id,
        "array_task_id": str(task),
        "seed": str(seed),
        "cluster": "hpc4",
        "account": "sigroup",
        "partition": "gpu-l20",
        "restart_count": "0",
        "phase2_design_sha256": result_value["phase2_design_sha256"],
        "base_config_hash": result_value["source_config_hash"],
        "git_commit": "a" * 40,
        "recovery_authorization_sha256": authorization_sha256,
        "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
        "submission_intent_sha256": PILOT_SUBMISSION_INTENT_SHA256,
        "submission_ledger_sha256": PILOT_SUBMISSION_LEDGER_SHA256,
        "materialization_mode": "fresh",
        "recovery_outputs_mounted": "false",
        "hf_root_mount_mode": "read_only",
        "datasets_cache_scope": "job_local",
        "artifact_metadata_sha256": _sha(run / "artifact" / "metadata.json"),
        "phase2_result_sha256": _sha(result),
        "phase2_output_verification_sha256": _sha(old_output),
        "post_recovery_output_verification_sha256": _sha(strict_output),
        "created_at_utc": "2026-07-25T00:00:00Z",
    }
    (run / "SUCCESS").write_text(
        "".join(f"{key}={value}\n" for key, value in marker.items()),
        encoding="utf-8",
        newline="\n",
    )


def _campaign(
    tmp_path: Path,
    *,
    authorization_sha256: str = "d" * 64,
    authorization_payload: dict[str, object] | None = None,
):
    pilot_helpers = _pilot_helpers()
    aggregate_helpers = _aggregate_helpers()
    config = aggregate_helpers._post_recovery_config()
    config["recovery_success_reference"]["artifact_sha256"] = authorization_sha256
    config["reward_model"]["optimizer_protocol"]["source_recovery_authorization_sha256"] = (
        authorization_sha256
    )
    if authorization_payload is not None:
        projection = config["recovery_success_reference"]["authorization_projection"]
        for field in (
            "recovery_design_sha256",
            "optimizer_schedule_sha256",
            "source_array_job_id",
            "execution_revision",
            "ordered_seeds",
            "recovery_status",
            "full_calibration_authorized",
            "authorized_information",
            "recovery_outputs_reusable",
            "validation_or_heldout_access_authorized",
            "policy_or_final_utility_access_authorized",
        ):
            projection[field] = authorization_payload[field]
    results = [
        pilot_helpers._seed_result(
            tmp_path,
            config,
            seed=seed,
            beta=beta,
        )
        for seed, beta in zip(
            (20260801, 20260802, 20260803),
            (1.5, 2.0, 1.75),
            strict=True,
        )
    ]
    for task, (seed, result) in enumerate(
        zip((20260801, 20260802, 20260803), results, strict=True)
    ):
        _upgrade_run(
            result.parent,
            helpers=aggregate_helpers,
            config=config,
            seed=seed,
            task=task,
            authorization_sha256=authorization_sha256,
        )
    return config, results


def _terminal() -> dict[str, object]:
    return {
        "rows": [
            {
                "job_id": f"1000_{task}",
                "job_id_raw": str(7000 + task),
                "array_job_id": "1000",
                "array_task_id": task,
                "seed": seed,
                "state": "COMPLETED",
                "exit_code": "0:0",
                "derived_exit_code": "0:0",
                "cluster": "hpc4",
                "account": "sigroup",
                "partition": "gpu-l20",
            }
            for task, seed in enumerate((20260801, 20260802, 20260803))
        ]
    }


def _patch_external_gates(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, object],
) -> None:
    monkeypatch.setattr(
        aggregate,
        "verify_recovery_authorization_config_binding",
        lambda *args, **kwargs: {
            "phase2_design_sha256": phase2_design_identity(config),
            "base_config_hash": config["design"]["source_config_hash"],
            "pilot_phase": config["design"]["pilot_phase"],
        },
    )
    monkeypatch.setattr(
        aggregate,
        "verify_post_recovery_terminal_evidence",
        lambda *args, **kwargs: _terminal(),
    )
    monkeypatch.setattr(
        aggregate,
        "load_phase2_config_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            config=config,
            design_identity=phase2_design_identity(config),
        ),
    )
    monkeypatch.setattr(
        aggregate,
        "_overlay_git_binding",
        lambda *args, **kwargs: {
            "phase2_overlay_repo_relative": (
                f"configs/common_beta_post_recovery_{config['design']['pilot_phase']}.yaml"
            ),
            "phase2_overlay_sha256": "9" * 64,
            "phase2_overlay_git_blob_sha1": "8" * 40,
            "phase2_overlay_git_commit": "a" * 40,
        },
    )


def test_three_seed_v3_aggregate_preserves_deep_selection_and_selects_maximum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, results = _campaign(tmp_path)
    _patch_external_gates(monkeypatch, config=config)

    payload = aggregate.build_phase2_post_recovery_aggregate(
        ROOT / "configs" / "common_beta_pilot.yaml",
        results,
        authorization_path=tmp_path / "authorization.json",
        authorization_sha256="d" * 64,
        terminal_evidence_path=tmp_path / "terminal.json",
        terminal_evidence_sha256="e" * 64,
        array_job_id="1000",
        submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
        submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
        submission_intent_reference_path=tmp_path / "submission-intent.json",
        submission_ledger_reference_path=tmp_path / "submission-ledger.json",
        aggregator_git_commit="f" * 40,
        producer_git_commit="a" * 40,
        image_sha256="b" * 64,
        hf_inventory_sha256="c" * 64,
        reference_base=tmp_path,
        phase2_overlay_reference_path=tmp_path / "aggregate.phase2-overlay.yaml",
    )

    assert payload["schema_version"] == "common-beta-pilot-selection-aggregate/v3"
    assert payload["selection"]["recommended_pilot_freeze_beta"] == 2.0
    assert payload["information_boundary"]["oracle_outcomes_consumed"] is False
    assert payload["post_recovery_control"] == {
        "schema_version": "phase2-post-recovery-aggregation-control/v1",
        "pilot_phase": "calibration",
        "phase2_overlay": "aggregate.phase2-overlay.yaml",
        "phase2_overlay_repo_relative": ("configs/common_beta_post_recovery_calibration.yaml"),
        "phase2_overlay_sha256": "9" * 64,
        "phase2_overlay_git_blob_sha1": "8" * 40,
        "phase2_overlay_git_commit": "a" * 40,
        "normalized_phase2_config": config,
        "normalized_phase2_config_sha256": phase2_design_identity(config),
        "recovery_authorization": "authorization.json",
        "recovery_authorization_sha256": "d" * 64,
        "optimizer_schedule_sha256": OPTIMIZER_SCHEDULE_SHA256,
        "submission_intent": "submission-intent.json",
        "submission_intent_sha256": PILOT_SUBMISSION_INTENT_SHA256,
        "submission_ledger": "submission-ledger.json",
        "submission_ledger_sha256": PILOT_SUBMISSION_LEDGER_SHA256,
        "pilot_terminal_evidence": "terminal.json",
        "pilot_terminal_evidence_sha256": "e" * 64,
        "pilot_array_job_id": "1000",
        "ordered_seeds": [20260801, 20260802, 20260803],
        "materialization_mode": "fresh",
        "recovery_outputs_reused": False,
        "all_tasks_terminal_completed_zero_exit": True,
        "post_recovery_validator_source_sha256": (aggregate._validator_source_sha256()),
        "phase2_deep_validator_source_sha256": (aggregate._deep_validator_source_sha256()),
    }
    assert [source["seed"] for source in payload["sources"]] == [
        20260801,
        20260802,
        20260803,
    ]
    assert all(source["post_recovery_output_verification_sha256"] for source in payload["sources"])
    for seed in (20260801, 20260802, 20260803):
        tail_receipt = payload["per_seed"][str(seed)]["repeated_label_tail_diagnostics"]
        assert tail_receipt["schema_version"] == "repeated-label-tail-diagnostics/v1"
        assert len(tail_receipt["diagnostics_sha256"]) == 64
        assert tail_receipt["scalar_only"] is True
        assert tail_receipt["descriptive_only"] is True
        assert tail_receipt["used_for_clipping"] is False
        assert tail_receipt["used_for_selection"] is False
        assert tail_receipt["used_for_gating"] is False
    assert "schema_version=prorm-phase2-post-recovery-pilot-run-status/v1" in (
        results[0].parent / "SUCCESS"
    ).read_text(encoding="utf-8")


def test_v3_builder_generalizes_freeze_and_forwards_predecessor_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate_helpers = _aggregate_helpers()
    pilot_helpers = _pilot_helpers()
    calibration = aggregate_helpers._post_recovery_config()
    freeze = pilot_helpers._freeze_config(
        calibration,
        beta=2.0,
        source_sha256="4" * 64,
    )
    freeze["design"]["name"] = "common-beta-post-recovery-freeze-v1"
    results = [
        pilot_helpers._seed_result(tmp_path, freeze, seed=seed, beta=2.0)
        for seed in (20260801, 20260802, 20260803)
    ]
    for task, (seed, result) in enumerate(
        zip((20260801, 20260802, 20260803), results, strict=True)
    ):
        _upgrade_run(
            result.parent,
            helpers=aggregate_helpers,
            config=freeze,
            seed=seed,
            task=task,
            authorization_sha256="d" * 64,
        )
    _patch_external_gates(monkeypatch, config=freeze)
    beta_source = tmp_path / "accepted-calibration-v3.json"
    horizon_parent = tmp_path / "calibration-horizon-v3.json"
    observed: dict[str, object] = {}

    def fake_legacy(
        config: dict[str, object],
        result_jsons: list[Path],
        **kwargs: object,
    ) -> dict[str, object]:
        observed["config"] = config
        observed["results"] = result_jsons
        observed.update(kwargs)
        return {
            "schema_version": "common-beta-pilot-selection-aggregate/v2",
            "pilot_phase": "freeze",
            "selection": {"beta_grid_index": 0},
        }

    monkeypatch.setattr(aggregate, "build_phase2_pilot_aggregate", fake_legacy)
    monkeypatch.setattr(
        aggregate,
        "verify_post_recovery_aggregate_success_receipt",
        lambda _: {"status": "verified"},
    )
    payload = aggregate.build_phase2_post_recovery_aggregate(
        ROOT / "configs" / "common_beta_pilot.yaml",
        results,
        authorization_path=tmp_path / "authorization.json",
        authorization_sha256="d" * 64,
        terminal_evidence_path=tmp_path / "terminal.json",
        terminal_evidence_sha256="e" * 64,
        array_job_id="1000",
        submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
        submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
        submission_intent_reference_path=tmp_path / "submission-intent.json",
        submission_ledger_reference_path=tmp_path / "submission-ledger.json",
        aggregator_git_commit="f" * 40,
        producer_git_commit="a" * 40,
        image_sha256="b" * 64,
        hf_inventory_sha256="c" * 64,
        reference_base=tmp_path,
        beta_source_aggregate_path=beta_source,
        horizon_parent_aggregate_path=horizon_parent,
    )

    assert payload["schema_version"] == "common-beta-pilot-selection-aggregate/v3"
    assert payload["pilot_phase"] == "freeze"
    assert payload["post_recovery_control"]["pilot_phase"] == "freeze"
    assert payload["post_recovery_control"]["phase2_overlay_repo_relative"] == (
        "configs/common_beta_post_recovery_freeze.yaml"
    )
    assert observed["beta_source_aggregate"] == beta_source
    assert observed["horizon_parent_aggregate"] == horizon_parent


@pytest.mark.parametrize(
    ("payload", "expected_overlay", "expected_aggregate"),
    [
        (
            {
                "pilot_phase": "calibration",
                "horizon": {"horizon_grid_index": 0},
            },
            "common_beta_post_recovery_calibration.yaml",
            "phase2-post-recovery-calibration-aggregate.json",
        ),
        (
            {
                "pilot_phase": "calibration",
                "horizon": {"horizon_grid_index": 2},
            },
            "common_beta_post_recovery_calibration_horizon_2.yaml",
            "phase2-post-recovery-calibration-horizon-2-aggregate.json",
        ),
        (
            {
                "pilot_phase": "freeze",
                "selection": {"beta_grid_index": 0},
            },
            "common_beta_post_recovery_freeze.yaml",
            "phase2-post-recovery-freeze-aggregate.json",
        ),
        (
            {
                "pilot_phase": "freeze",
                "selection": {"beta_grid_index": 3},
            },
            "common_beta_post_recovery_freeze_retry_3.yaml",
            "phase2-post-recovery-freeze-retry-3-aggregate.json",
        ),
    ],
)
def test_v3_publication_filename_is_semantic_lineage(
    payload: dict[str, object],
    expected_overlay: str,
    expected_aggregate: str,
) -> None:
    assert aggregate._semantic_lineage_filenames(payload) == (
        expected_overlay,
        expected_aggregate,
    )
    assert aggregate._semantic_aggregate_filename(payload) == expected_aggregate


@pytest.mark.parametrize(
    "payload",
    [
        {"pilot_phase": "calibration", "horizon": {"horizon_grid_index": True}},
        {"pilot_phase": "freeze", "selection": {"beta_grid_index": -1}},
        {"pilot_phase": "freeze", "selection": {}},
        {"pilot_phase": "confirmatory"},
    ],
)
def test_v3_semantic_lineage_rejects_missing_or_forged_indices(
    payload: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError), match="grid index|pilot phase"):
        aggregate._semantic_lineage_filenames(payload)


def test_v3_reader_derives_exact_overlay_and_aggregate_names_from_design() -> None:
    aggregate_helpers = _aggregate_helpers()
    pilot_helpers = _pilot_helpers()
    calibration = aggregate_helpers._post_recovery_config()
    calibration_design = pilot_reader.Phase2Design.from_phase2_config(calibration)
    assert pilot_reader._post_recovery_semantic_filenames(
        calibration_design,
        source_binding=None,
    ) == (
        "common_beta_post_recovery_calibration.yaml",
        "phase2-post-recovery-calibration-aggregate.json",
    )

    escalated = pilot_helpers._escalated_calibration_config(
        calibration,
        horizon=512,
        horizon_grid_index=1,
        parent_sha256="1" * 64,
    )
    escalated_design = pilot_reader.Phase2Design.from_phase2_config(escalated)
    assert pilot_reader._post_recovery_semantic_filenames(
        escalated_design,
        source_binding=None,
    ) == (
        "common_beta_post_recovery_calibration_horizon_1.yaml",
        "phase2-post-recovery-calibration-horizon-1-aggregate.json",
    )

    freeze = pilot_helpers._freeze_config(
        calibration,
        beta=2.0,
        source_sha256="2" * 64,
    )
    freeze_design = pilot_reader.Phase2Design.from_phase2_config(freeze)
    assert pilot_reader._post_recovery_semantic_filenames(
        freeze_design,
        source_binding={"beta_grid_index": 0},
    ) == (
        "common_beta_post_recovery_freeze.yaml",
        "phase2-post-recovery-freeze-aggregate.json",
    )
    assert pilot_reader._post_recovery_semantic_filenames(
        freeze_design,
        source_binding={"beta_grid_index": 2},
    ) == (
        "common_beta_post_recovery_freeze_retry_2.yaml",
        "phase2-post-recovery-freeze-retry-2-aggregate.json",
    )


def test_v3_deep_lineage_horizon_retry_and_confirmatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path
    recovery = _recovery_helpers()
    recovery_paths, scheduler, authorization, _ = recovery._campaign(
        project,
        monkeypatch,
    )
    recovery.aggregate.write_phase2_recovery_authorization(
        recovery_paths,
        authorization,
        scheduler_evidence=scheduler,
        aggregator_git_commit=recovery.AGGREGATOR_COMMIT,
    )
    authorization_sha256 = _sha(authorization)
    authorization_payload = json.loads(authorization.read_text(encoding="utf-8"))
    if authorization_payload["recovery_design_sha256"] != (
        phase2_config_module._RECOVERY_DESIGN_SHA256
    ):
        monkeypatch.setattr(
            phase2_config_module,
            "_RECOVERY_DESIGN_SHA256",
            authorization_payload["recovery_design_sha256"],
        )

    aggregate_helpers = _aggregate_helpers()
    pilot_helpers = _pilot_helpers()
    config_helpers = _config_helpers()
    initial = aggregate_helpers._post_recovery_config()
    initial["recovery_success_reference"]["artifact_sha256"] = authorization_sha256
    initial["reward_model"]["optimizer_protocol"]["source_recovery_authorization_sha256"] = (
        authorization_sha256
    )
    projection = initial["recovery_success_reference"]["authorization_projection"]
    for field in (
        "recovery_design_sha256",
        "optimizer_schedule_sha256",
        "source_array_job_id",
        "execution_revision",
        "ordered_seeds",
        "recovery_status",
        "full_calibration_authorized",
        "authorized_information",
        "recovery_outputs_reusable",
        "validation_or_heldout_access_authorized",
        "policy_or_final_utility_access_authorized",
    ):
        projection[field] = authorization_payload[field]

    monkeypatch.setattr(pilot_reader, "_POST_RECOVERY_PROJECT_ROOT", project)
    monkeypatch.setattr(
        pilot_reader,
        "verify_post_recovery_aggregate_success_receipt",
        lambda _: {"status": "verified-scientific-reader-fixture"},
    )
    monkeypatch.setattr(
        pilot_reader,
        "verify_post_recovery_submission_evidence",
        lambda *args, **kwargs: {"status": "verified-scientific-reader-fixture"},
    )
    monkeypatch.setattr(
        aggregate,
        "verify_post_recovery_aggregate_success_receipt",
        lambda _: {"status": "verified-scientific-reader-fixture"},
    )
    aggregate_root = project / "aggregates"
    aggregate_root.mkdir()
    real_subprocess_run = subprocess.run
    git_objects = {
        f"{'f' * 40}:src/smart_reward/phase2_pilot_aggregate.py": (
            Path(pilot_reader.__file__).read_bytes()
        ),
        f"{'f' * 40}:src/smart_reward/phase2_aggregate.py": (
            (ROOT / "src" / "smart_reward" / "phase2_aggregate.py").read_bytes()
        ),
        f"{'f' * 40}:src/smart_reward/phase2_post_recovery_aggregate.py": (
            Path(aggregate.__file__).read_bytes()
        ),
    }

    def git_show(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert arguments[0] == "git" and arguments[-2] == "show"
        assert arguments[-1] in git_objects
        return subprocess.CompletedProcess(
            args=arguments,
            returncode=0,
            stdout=git_objects[arguments[-1]],
            stderr=b"",
        )

    def fake_overlay_git_binding(
        raw_overlay_path: str | os.PathLike[str],
        **_: object,
    ) -> dict[str, str]:
        raw = Path(raw_overlay_path).read_bytes()
        return {
            "phase2_overlay_repo_relative": f"configs/{Path(raw_overlay_path).name}",
            "phase2_overlay_sha256": hashlib.sha256(raw).hexdigest(),
            "phase2_overlay_git_blob_sha1": hashlib.sha1(
                f"blob {len(raw)}\0".encode("ascii") + raw
            ).hexdigest(),
            "phase2_overlay_git_commit": "a" * 40,
        }

    monkeypatch.setattr(aggregate, "_overlay_git_binding", fake_overlay_git_binding)

    def publish(
        config: dict[str, object],
        *,
        array_job_id: str,
        raw_job_id_base: int,
        betas: tuple[float, float, float],
        beta_grid_index: int | None,
        beta_source: Path | None = None,
        horizon_parent: Path | None = None,
        unsafe: bool = False,
        length_unsafe: bool = False,
    ) -> tuple[Path, dict[str, object]]:
        phase = str(config["design"]["pilot_phase"])
        campaign_root = project / "runs" / f"phase2-post-recovery-{phase}"
        campaign_root.mkdir(parents=True, exist_ok=True)
        results: list[Path] = []
        for task, (seed, beta) in enumerate(
            zip((20260801, 20260802, 20260803), betas, strict=True)
        ):
            result = pilot_helpers._seed_result(
                campaign_root,
                config,
                seed=seed,
                beta=beta,
                unsafe=unsafe,
                length_unsafe=length_unsafe,
            )
            renamed_run = result.parent.parent / f"job-{array_job_id}_{task}"
            result.parent.rename(renamed_run)
            result = renamed_run / result.name
            _upgrade_run(
                renamed_run,
                helpers=aggregate_helpers,
                config=config,
                seed=seed,
                task=task,
                authorization_sha256=authorization_sha256,
                array_job_id=array_job_id,
                raw_job_id_base=raw_job_id_base,
            )
            results.append(result)

        design_sha256 = phase2_design_identity(config)
        monkeypatch.setattr(subprocess, "run", real_subprocess_run)
        for task, (seed, result) in enumerate(
            zip((20260801, 20260802, 20260803), results, strict=True)
        ):
            artifact_link = result.parent / "artifact"
            artifact_target = (
                project
                / "artifacts"
                / f"phase2-post-recovery-{phase}"
                / design_sha256
                / f"seed-{seed}"
                / f"job-{array_job_id}_{task}"
            )
            artifact_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(artifact_link, artifact_target)
            shutil.rmtree(artifact_link)
            _link_directory(artifact_link, artifact_target)

        raw_sacct = "".join(
            f"{array_job_id}_{task}|{raw_job_id_base + task}|COMPLETED|0:0|0:0|"
            "hpc4|sigroup|gpu-l20|1|8|"
            "billing=8,cpu=8,gres/gpu=1,mem=96G,node=1|"
            "billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1\n"
            for task in range(3)
        ).encode()
        terminal = campaign_root / f"terminal-{array_job_id}.json"
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=0,
                stdout=raw_sacct,
                stderr=b"",
            ),
        )
        control.capture_post_recovery_terminal_evidence(
            array_job_id,
            terminal,
            pilot_phase=phase,
        )

        design = pilot_reader.Phase2Design.from_phase2_config(config)
        source_binding = None if beta_grid_index is None else {"beta_grid_index": beta_grid_index}
        overlay_name, aggregate_name = pilot_reader._post_recovery_semantic_filenames(
            design,
            source_binding=source_binding,
        )
        output = aggregate_root / aggregate_name
        evidence_root = Path(f"{output}.evidence")
        evidence_configs = evidence_root / "configs"
        evidence_configs.mkdir(parents=True)
        overlay = evidence_configs / overlay_name
        overlay.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        (evidence_configs / "common_beta_pilot_base.yaml").write_bytes(
            (ROOT / "configs" / "common_beta_pilot_base.yaml").read_bytes()
        )
        overlay_raw = overlay.read_bytes()
        git_objects[f"{'a' * 40}:configs/{overlay_name}"] = overlay_raw
        monkeypatch.setattr(subprocess, "run", git_show)
        payload = aggregate.write_phase2_post_recovery_aggregate(
            overlay,
            results,
            output,
            authorization_path=authorization,
            authorization_sha256=authorization_sha256,
            terminal_evidence_path=terminal,
            terminal_evidence_sha256=_sha(terminal),
            array_job_id=array_job_id,
            submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
            submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
            submission_intent_reference_path=(
                evidence_root / "submission-registry" / "intent.json"
            ),
            submission_ledger_reference_path=(
                evidence_root / "submission-registry" / "submission.json"
            ),
            aggregator_git_commit="f" * 40,
            producer_git_commit="a" * 40,
            image_sha256="b" * 64,
            hf_inventory_sha256="c" * 64,
            reference_base=aggregate_root,
            phase2_overlay_reference_path=overlay,
            beta_source_aggregate_path=beta_source,
            horizon_parent_aggregate_path=horizon_parent,
        )
        return output, payload

    calibration_0, calibration_0_payload = publish(
        initial,
        array_job_id="1100",
        raw_job_id_base=7100,
        betas=(1.5, 2.0, 1.75),
        beta_grid_index=None,
        length_unsafe=True,
    )
    assert calibration_0_payload["horizon"]["all_seed_length_gates_passed"] is False
    assert calibration_0_payload["selection"]["next_horizon_tokens"] == 512
    premature_freeze = pilot_helpers._freeze_config(
        initial,
        beta=float(calibration_0_payload["selection"]["recommended_pilot_freeze_beta"]),
        source_sha256=_sha(calibration_0),
    )
    premature_freeze["design"]["name"] = "common-beta-post-recovery-premature-freeze"
    with pytest.raises(ValueError, match="did not accept all length gates"):
        pilot_reader.verify_horizon_parent_aggregate(
            premature_freeze,
            calibration_0,
        )

    calibration_1_config = pilot_helpers._escalated_calibration_config(
        initial,
        horizon=512,
        horizon_grid_index=1,
        parent_sha256=_sha(calibration_0),
    )
    calibration_1_config["design"]["name"] = "common-beta-post-recovery-calibration-horizon-1"
    calibration_1, calibration_1_payload = publish(
        calibration_1_config,
        array_job_id="1200",
        raw_job_id_base=7200,
        betas=(1.5, 2.0, 1.75),
        beta_grid_index=None,
        horizon_parent=calibration_0,
    )
    assert calibration_1_payload["horizon"]["all_seed_length_gates_passed"] is True
    frozen_beta = float(calibration_1_payload["selection"]["recommended_pilot_freeze_beta"])

    freeze_0_config = pilot_helpers._freeze_config(
        calibration_1_config,
        beta=frozen_beta,
        source_sha256=_sha(calibration_1),
    )
    freeze_0_config["design"]["name"] = "common-beta-post-recovery-freeze-0"
    freeze_0, freeze_0_payload = publish(
        freeze_0_config,
        array_job_id="1300",
        raw_job_id_base=7300,
        betas=(frozen_beta, frozen_beta, frozen_beta),
        beta_grid_index=0,
        beta_source=calibration_1,
        horizon_parent=calibration_1,
        unsafe=True,
    )
    assert freeze_0_payload["selection"]["selection_accepted"] is False
    assert freeze_0_payload["selection"]["next_global_beta"] == 2.0 * frozen_beta

    retry_beta = 2.0 * frozen_beta
    retry_config = pilot_helpers._freeze_config(
        calibration_1_config,
        beta=retry_beta,
        source_sha256=_sha(freeze_0),
        horizon_parent_sha256=_sha(calibration_1),
    )
    retry_config["design"]["name"] = "common-beta-post-recovery-freeze-retry-1"
    retry, retry_payload = publish(
        retry_config,
        array_job_id="1400",
        raw_job_id_base=7400,
        betas=(retry_beta, retry_beta, retry_beta),
        beta_grid_index=1,
        beta_source=freeze_0,
        horizon_parent=calibration_1,
    )
    assert retry_payload["selection"]["beta_grid_index"] == 1
    assert retry_payload["selection"]["selection_accepted"] is True
    assert retry_payload["selection"]["accepted_for_confirmatory_identity"] is True

    confirmatory_overlay, confirmatory_base = config_helpers._future_confirmatory(retry_config)
    confirmatory_overlay["design"]["name"] = "common-beta-post-recovery-confirmatory-deep-lineage"
    confirmatory_overlay["objective"]["common_beta"]["frozen_global_beta"] = retry_beta
    confirmatory_overlay["objective"]["common_beta"]["beta_source_aggregate_sha256"] = _sha(retry)
    confirmatory_overlay["evaluation"]["max_length"]["parent_pilot_aggregate_sha256"] = _sha(retry)
    confirmatory = validate_phase2_config(
        confirmatory_overlay,
        base_config=confirmatory_base,
    )
    beta_binding = pilot_reader.verify_beta_source_aggregate(confirmatory, retry)
    horizon_binding = pilot_reader.verify_horizon_parent_aggregate(confirmatory, retry)
    assert beta_binding is not None and beta_binding["accepted_beta"] == retry_beta
    assert horizon_binding is not None
    assert horizon_binding["source_pilot_phase"] == "freeze"
    assert confirmatory["design"]["stage"] == "confirmatory"
    assert all(
        payload["post_recovery_control"]["recovery_authorization_sha256"] == authorization_sha256
        for payload in (
            calibration_0_payload,
            calibration_1_payload,
            freeze_0_payload,
            retry_payload,
        )
    )


def test_v3_aggregate_rejects_tampered_freshness_marker_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, results = _campaign(tmp_path)
    _patch_external_gates(monkeypatch, config=config)
    marker = results[0].parent / "SUCCESS"
    marker.write_text(
        marker.read_text(encoding="utf-8").replace(
            "materialization_mode=fresh",
            "materialization_mode=reused",
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="materialization_mode"):
        aggregate.build_phase2_post_recovery_aggregate(
            ROOT / "configs" / "common_beta_pilot.yaml",
            results,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="d" * 64,
            terminal_evidence_path=tmp_path / "terminal.json",
            terminal_evidence_sha256="e" * 64,
            array_job_id="1000",
            submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
            submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
            submission_intent_reference_path=tmp_path / "submission-intent.json",
            submission_ledger_reference_path=tmp_path / "submission-ledger.json",
            aggregator_git_commit="f" * 40,
            producer_git_commit="a" * 40,
            image_sha256="b" * 64,
            hf_inventory_sha256="c" * 64,
            reference_base=tmp_path,
            phase2_overlay_reference_path=(tmp_path / "aggregate.phase2-overlay.yaml"),
        )


def test_v3_aggregate_rejects_success_terminal_allocation_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, results = _campaign(tmp_path)
    _patch_external_gates(monkeypatch, config=config)
    marker = results[0].parent / "SUCCESS"
    marker.write_text(
        marker.read_text(encoding="utf-8")
        .replace("slurm_job_id=7000", "slurm_job_id=7999")
        .replace("allocation_job_id_raw=7000", "allocation_job_id_raw=7999"),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="slurm_job_id"):
        aggregate.build_phase2_post_recovery_aggregate(
            ROOT / "configs" / "common_beta_pilot.yaml",
            results,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="d" * 64,
            terminal_evidence_path=tmp_path / "terminal.json",
            terminal_evidence_sha256="e" * 64,
            array_job_id="1000",
            submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
            submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
            submission_intent_reference_path=tmp_path / "submission-intent.json",
            submission_ledger_reference_path=tmp_path / "submission-ledger.json",
            aggregator_git_commit="f" * 40,
            producer_git_commit="a" * 40,
            image_sha256="b" * 64,
            hf_inventory_sha256="c" * 64,
            reference_base=tmp_path,
        )


def test_v3_aggregate_rejects_duplicate_terminal_allocation_job_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, results = _campaign(tmp_path)
    _patch_external_gates(monkeypatch, config=config)
    terminal = _terminal()
    terminal["rows"][1]["job_id_raw"] = "7000"
    monkeypatch.setattr(
        aggregate,
        "verify_post_recovery_terminal_evidence",
        lambda *args, **kwargs: terminal,
    )

    with pytest.raises(ValueError, match="JobIDRaw values must be unique"):
        aggregate.build_phase2_post_recovery_aggregate(
            ROOT / "configs" / "common_beta_pilot.yaml",
            results,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="d" * 64,
            terminal_evidence_path=tmp_path / "terminal.json",
            terminal_evidence_sha256="e" * 64,
            array_job_id="1000",
            submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
            submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
            submission_intent_reference_path=tmp_path / "submission-intent.json",
            submission_ledger_reference_path=tmp_path / "submission-ledger.json",
            aggregator_git_commit="f" * 40,
            producer_git_commit="a" * 40,
            image_sha256="b" * 64,
            hf_inventory_sha256="c" * 64,
            reference_base=tmp_path,
        )


def test_v3_aggregate_stops_on_authorization_or_terminal_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, results = _campaign(tmp_path)
    monkeypatch.setattr(
        aggregate,
        "verify_recovery_authorization_config_binding",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("authorization payload projection mismatch")
        ),
    )
    with pytest.raises(ValueError, match="authorization"):
        aggregate.build_phase2_post_recovery_aggregate(
            ROOT / "configs" / "common_beta_pilot.yaml",
            results,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="d" * 64,
            terminal_evidence_path=tmp_path / "terminal.json",
            terminal_evidence_sha256="e" * 64,
            array_job_id="1000",
            submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
            submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
            submission_intent_reference_path=tmp_path / "submission-intent.json",
            submission_ledger_reference_path=tmp_path / "submission-ledger.json",
            aggregator_git_commit="f" * 40,
            producer_git_commit="a" * 40,
            image_sha256="b" * 64,
            hf_inventory_sha256="c" * 64,
            reference_base=tmp_path,
        )


def test_seed_output_receipt_uses_the_public_deep_five_head_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, results = _campaign(tmp_path)
    result = results[0]
    run = result.parent
    old_receipt = run / "phase2-output-verification.json"
    strict_receipt = run / "post-recovery-output-verification.json"
    old_receipt.unlink()
    strict_receipt.unlink()
    design_sha256 = phase2_design_identity(config)
    monkeypatch.setattr(
        output_verifier,
        "verify_recovery_authorization_config_binding",
        lambda *args, **kwargs: {
            "phase2_design_sha256": design_sha256,
            "base_config_hash": config["design"]["source_config_hash"],
        },
    )
    monkeypatch.setattr(
        output_verifier,
        "load_phase2_config_bundle",
        lambda *args, **kwargs: SimpleNamespace(config=config),
    )

    receipt = output_verifier.verify_and_write_post_recovery_output(
        overlay_path=tmp_path / "overlay.yaml",
        authorization_path=tmp_path / "authorization.json",
        authorization_sha256="d" * 64,
        result_path=result,
        diagnostics_path=run / "phase2-pilot-diagnostics.diagnostics.jsonl",
        phase2_output_verification_path=old_receipt,
        post_recovery_output_verification_path=strict_receipt,
        seed=20260801,
        expected_design_sha256=design_sha256,
        expected_base_config_hash=config["design"]["source_config_hash"],
        expected_git_commit="a" * 40,
        expected_image_sha256="b" * 64,
        expected_hf_inventory_sha256="c" * 64,
        expected_artifact_metadata_sha256=_sha(run / "artifact" / "metadata.json"),
        expected_slurm_job_id_raw="7000",
        expected_array_job_id="1000",
        expected_array_task_id=0,
        expected_pilot_phase="calibration",
    )

    assert receipt["five_head_adopted_schedule_verified"] is True
    assert set(receipt["five_head_training"]) == {
        "primary_bt_mle",
        "primary_prorm_plus",
        "low_dimensional_prorm_plus",
        "exact_margin_prorm_plus",
        "exact_soft_label_bt",
    }
    assert all(
        item["schedule_sha256"] == OPTIMIZER_SCHEDULE_SHA256
        for item in receipt["five_head_training"].values()
    )
    assert json.loads(strict_receipt.read_text(encoding="utf-8")) == receipt


def test_seed_output_receipts_roll_back_legacy_receipt_if_strict_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, results = _campaign(tmp_path)
    result = results[0]
    run = result.parent
    old_receipt = run / "phase2-output-verification.json"
    strict_receipt = run / "post-recovery-output-verification.json"
    old_receipt.unlink()
    strict_receipt.unlink()
    design_sha256 = phase2_design_identity(config)
    monkeypatch.setattr(
        output_verifier,
        "verify_recovery_authorization_config_binding",
        lambda *args, **kwargs: {
            "phase2_design_sha256": design_sha256,
            "base_config_hash": config["design"]["source_config_hash"],
        },
    )
    monkeypatch.setattr(
        output_verifier,
        "load_phase2_config_bundle",
        lambda *args, **kwargs: SimpleNamespace(config=config),
    )
    real_atomic_write_json = output_verifier.atomic_write_json
    publications = 0

    def fail_second_publication(
        path: str | os.PathLike[str],
        value: dict[str, object],
        *,
        overwrite: bool,
    ) -> None:
        nonlocal publications
        publications += 1
        if publications == 2:
            raise OSError("injected strict receipt publication failure")
        real_atomic_write_json(path, value, overwrite=overwrite)

    monkeypatch.setattr(
        output_verifier,
        "atomic_write_json",
        fail_second_publication,
    )

    with pytest.raises(OSError, match="injected strict receipt"):
        output_verifier.verify_and_write_post_recovery_output(
            overlay_path=tmp_path / "overlay.yaml",
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="d" * 64,
            result_path=result,
            diagnostics_path=run / "phase2-pilot-diagnostics.diagnostics.jsonl",
            phase2_output_verification_path=old_receipt,
            post_recovery_output_verification_path=strict_receipt,
            seed=20260801,
            expected_design_sha256=design_sha256,
            expected_base_config_hash=config["design"]["source_config_hash"],
            expected_git_commit="a" * 40,
            expected_image_sha256="b" * 64,
            expected_hf_inventory_sha256="c" * 64,
            expected_artifact_metadata_sha256=_sha(run / "artifact" / "metadata.json"),
            expected_slurm_job_id_raw="7000",
            expected_array_job_id="1000",
            expected_array_task_id=0,
            expected_pilot_phase="calibration",
        )

    assert publications == 2
    assert not old_receipt.exists()
    assert not strict_receipt.exists()


@pytest.mark.parametrize("attack", ["duplicate_key", "nested_vector"])
def test_seed_output_receipt_rejects_strict_sidecar_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    config, results = _campaign(tmp_path)
    result = results[0]
    run = result.parent
    old_receipt = run / "phase2-output-verification.json"
    strict_receipt = run / "post-recovery-output-verification.json"
    old_receipt.unlink()
    strict_receipt.unlink()
    diagnostics = run / "phase2-pilot-diagnostics.diagnostics.jsonl"
    lines = diagnostics.read_text(encoding="utf-8").splitlines()
    if attack == "duplicate_key":
        lines[0] = lines[0].replace(
            '{"arm":',
            '{"arm":"zero_b","arm":',
            1,
        )
    else:
        row = json.loads(lines[0])
        row["nested_attack"] = {"audit": {"head_weight": [0.0]}}
        lines[0] = json.dumps(
            row,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    diagnostics.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result_value = json.loads(result.read_text(encoding="utf-8"))
    result_value["diagnostics_sha256"] = _sha(diagnostics)
    _write_json(result, result_value)
    design_sha256 = phase2_design_identity(config)
    monkeypatch.setattr(
        output_verifier,
        "verify_recovery_authorization_config_binding",
        lambda *args, **kwargs: {
            "phase2_design_sha256": design_sha256,
            "base_config_hash": config["design"]["source_config_hash"],
        },
    )
    monkeypatch.setattr(
        output_verifier,
        "load_phase2_config_bundle",
        lambda *args, **kwargs: SimpleNamespace(config=config),
    )

    with pytest.raises(ValueError, match="strict JSON|target-free"):
        output_verifier.verify_and_write_post_recovery_output(
            overlay_path=tmp_path / "overlay.yaml",
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="d" * 64,
            result_path=result,
            diagnostics_path=diagnostics,
            phase2_output_verification_path=old_receipt,
            post_recovery_output_verification_path=strict_receipt,
            seed=20260801,
            expected_design_sha256=design_sha256,
            expected_base_config_hash=config["design"]["source_config_hash"],
            expected_git_commit="a" * 40,
            expected_image_sha256="b" * 64,
            expected_hf_inventory_sha256="c" * 64,
            expected_artifact_metadata_sha256=_sha(run / "artifact" / "metadata.json"),
            expected_slurm_job_id_raw="7000",
            expected_array_job_id="1000",
            expected_array_task_id=0,
            expected_pilot_phase="calibration",
        )


def test_v3_builder_rejects_duplicate_key_in_strict_output_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, results = _campaign(tmp_path)
    _patch_external_gates(monkeypatch, config=config)
    strict = results[0].parent / "post-recovery-output-verification.json"
    raw = strict.read_text(encoding="utf-8")
    status_token = '"status":"passed"' if '"status":"passed"' in raw else '"status": "passed"'
    duplicate_status = (
        '"status":"passed","status":"passed"'
        if ":" in status_token and ": " not in status_token
        else '"status": "passed", "status": "passed"'
    )
    strict.write_text(
        raw.replace(status_token, duplicate_status, 1),
        encoding="utf-8",
        newline="\n",
    )
    marker_path = results[0].parent / "SUCCESS"
    marker: dict[str, str] = {}
    for line in marker_path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        marker[key] = value
    marker["post_recovery_output_verification_sha256"] = _sha(strict)
    marker_path.write_text(
        "".join(f"{key}={value}\n" for key, value in marker.items()),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        aggregate.build_phase2_post_recovery_aggregate(
            ROOT / "configs" / "common_beta_pilot.yaml",
            results,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="d" * 64,
            terminal_evidence_path=tmp_path / "terminal.json",
            terminal_evidence_sha256="e" * 64,
            array_job_id="1000",
            submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
            submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
            submission_intent_reference_path=tmp_path / "submission-intent.json",
            submission_ledger_reference_path=tmp_path / "submission-ledger.json",
            aggregator_git_commit="f" * 40,
            producer_git_commit="a" * 40,
            image_sha256="b" * 64,
            hf_inventory_sha256="c" * 64,
            reference_base=tmp_path,
        )


def test_v3_builder_rejects_coherently_rehashed_strict_diagnostics_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, results = _campaign(tmp_path)
    _patch_external_gates(monkeypatch, config=config)
    strict = results[0].parent / "post-recovery-output-verification.json"
    strict_value = json.loads(strict.read_text(encoding="utf-8"))
    assert strict_value["diagnostics_sha256"] == _sha(
        results[0].parent / "phase2-pilot-diagnostics.diagnostics.jsonl"
    )
    strict_value["diagnostics_sha256"] = "0" * 64
    _write_json(strict, strict_value)

    marker_path = results[0].parent / "SUCCESS"
    marker: dict[str, str] = {}
    for line in marker_path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        marker[key] = value
    marker["post_recovery_output_verification_sha256"] = _sha(strict)
    marker_path.write_text(
        "".join(f"{key}={value}\n" for key, value in marker.items()),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="output verification identity"):
        aggregate.build_phase2_post_recovery_aggregate(
            ROOT / "configs" / "common_beta_pilot.yaml",
            results,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="d" * 64,
            terminal_evidence_path=tmp_path / "terminal.json",
            terminal_evidence_sha256="e" * 64,
            array_job_id="1000",
            submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
            submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
            submission_intent_reference_path=tmp_path / "submission-intent.json",
            submission_ledger_reference_path=tmp_path / "submission-ledger.json",
            aggregator_git_commit="f" * 40,
            producer_git_commit="a" * 40,
            image_sha256="b" * 64,
            hf_inventory_sha256="c" * 64,
            reference_base=tmp_path,
        )


def test_written_v3_is_consumable_and_deep_tamper_survives_no_rehash_attack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path
    recovery = _recovery_helpers()
    recovery_paths, scheduler, authorization, _ = recovery._campaign(
        project,
        monkeypatch,
    )
    recovery.aggregate.write_phase2_recovery_authorization(
        recovery_paths,
        authorization,
        scheduler_evidence=scheduler,
        aggregator_git_commit=recovery.AGGREGATOR_COMMIT,
    )
    authorization_sha256 = _sha(authorization)
    authorization_payload = json.loads(authorization.read_text(encoding="utf-8"))
    if authorization_payload["recovery_design_sha256"] != (
        phase2_config_module._RECOVERY_DESIGN_SHA256
    ):
        monkeypatch.setattr(
            phase2_config_module,
            "_RECOVERY_DESIGN_SHA256",
            authorization_payload["recovery_design_sha256"],
        )

    terminal_root = project / "runs" / "phase2-post-recovery-calibration"
    terminal_root.mkdir(parents=True, exist_ok=True)
    terminal = terminal_root / "terminal-1000.json"
    raw_sacct = "".join(
        f"1000_{task}|{7000 + task}|COMPLETED|0:0|0:0|hpc4|sigroup|"
        "gpu-l20|1|8|billing=8,cpu=8,gres/gpu=1,mem=96G,node=1|"
        "billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1\n"
        for task in range(3)
    ).encode()
    real_subprocess_run = subprocess.run
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=raw_sacct,
            stderr=b"",
        ),
    )
    control.capture_post_recovery_terminal_evidence(
        "1000",
        terminal,
        pilot_phase="calibration",
    )
    monkeypatch.setattr(subprocess, "run", real_subprocess_run)
    terminal_sha256 = _sha(terminal)

    config, results = _campaign(
        terminal_root,
        authorization_sha256=authorization_sha256,
        authorization_payload=authorization_payload,
    )
    design_sha256 = phase2_design_identity(config)
    for task, (seed, result) in enumerate(
        zip((20260801, 20260802, 20260803), results, strict=True)
    ):
        run = result.parent
        artifact_link = run / "artifact"
        artifact_target = (
            project
            / "artifacts"
            / "phase2-post-recovery-calibration"
            / design_sha256
            / f"seed-{seed}"
            / f"job-1000_{task}"
        )
        artifact_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(artifact_link, artifact_target)
        shutil.rmtree(artifact_link)
        _link_directory(artifact_link, artifact_target)
    monkeypatch.setattr(pilot_reader, "_POST_RECOVERY_PROJECT_ROOT", project)

    def verify_scientific_reader_fixture(path: Path) -> dict[str, str]:
        if Path(path).resolve().parent != (project / "aggregates").resolve():
            raise AssertionError("publication verifier called outside the locked namespace")
        return {"status": "verified-scientific-reader-fixture"}

    monkeypatch.setattr(
        pilot_reader,
        "verify_post_recovery_aggregate_success_receipt",
        verify_scientific_reader_fixture,
    )
    monkeypatch.setattr(
        pilot_reader,
        "verify_post_recovery_submission_evidence",
        lambda *args, **kwargs: {"status": "verified-scientific-reader-fixture"},
    )
    monkeypatch.setattr(
        aggregate,
        "verify_post_recovery_aggregate_success_receipt",
        lambda _: {"status": "verified-scientific-reader-fixture"},
    )
    output_root = project / "aggregates"
    output_root.mkdir()
    aggregate_path = output_root / "phase2-post-recovery-calibration-aggregate.json"
    evidence_root = Path(f"{aggregate_path}.evidence")
    final_evidence_root = evidence_root
    evidence_configs = evidence_root / "configs"
    evidence_configs.mkdir(parents=True)
    overlay = evidence_configs / "common_beta_post_recovery_calibration.yaml"
    overlay.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    base = evidence_configs / "common_beta_pilot_base.yaml"
    base.write_bytes((ROOT / "configs" / "common_beta_pilot_base.yaml").read_bytes())
    overlay_raw = overlay.read_bytes()

    def fake_overlay_git_binding(
        raw_overlay_path: str | os.PathLike[str],
        **_: object,
    ) -> dict[str, str]:
        raw = Path(raw_overlay_path).read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
        return {
            "phase2_overlay_repo_relative": f"configs/{Path(raw_overlay_path).name}",
            "phase2_overlay_sha256": hashlib.sha256(raw).hexdigest(),
            "phase2_overlay_git_blob_sha1": blob,
            "phase2_overlay_git_commit": "a" * 40,
        }

    monkeypatch.setattr(
        aggregate,
        "_overlay_git_binding",
        fake_overlay_git_binding,
    )
    written = aggregate.write_phase2_post_recovery_aggregate(
        overlay,
        results,
        aggregate_path,
        authorization_path=authorization,
        authorization_sha256=authorization_sha256,
        terminal_evidence_path=terminal,
        terminal_evidence_sha256=terminal_sha256,
        array_job_id="1000",
        submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
        submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
        submission_intent_reference_path=(evidence_root / "submission-registry" / "intent.json"),
        submission_ledger_reference_path=(
            evidence_root / "submission-registry" / "submission.json"
        ),
        aggregator_git_commit="f" * 40,
        producer_git_commit="a" * 40,
        image_sha256="b" * 64,
        hf_inventory_sha256="c" * 64,
        reference_base=output_root,
        phase2_overlay_reference_path=overlay,
    )
    assert json.loads(aggregate_path.read_text(encoding="utf-8")) == written
    control_payload = written["post_recovery_control"]
    assert (output_root / control_payload["phase2_overlay"]).resolve() == (
        final_evidence_root / "configs" / "common_beta_post_recovery_calibration.yaml"
    )
    assert (output_root / control_payload["recovery_authorization"]).resolve() == (authorization)
    assert (output_root / control_payload["pilot_terminal_evidence"]).resolve() == terminal

    pilot_helpers = _pilot_helpers()
    aggregate_helpers = _aggregate_helpers()
    git_objects = {
        f"{'a' * 40}:configs/common_beta_post_recovery_calibration.yaml": (overlay_raw),
        f"{'f' * 40}:src/smart_reward/phase2_pilot_aggregate.py": (
            Path(pilot_reader.__file__).read_bytes()
        ),
        f"{'f' * 40}:src/smart_reward/phase2_aggregate.py": (
            (ROOT / "src" / "smart_reward" / "phase2_aggregate.py").read_bytes()
        ),
        f"{'f' * 40}:src/smart_reward/phase2_post_recovery_aggregate.py": (
            Path(aggregate.__file__).read_bytes()
        ),
    }

    def git_show(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert arguments[0] == "git" and arguments[-2] == "show"
        assert arguments[-1] in git_objects
        return subprocess.CompletedProcess(
            args=arguments,
            returncode=0,
            stdout=git_objects[arguments[-1]],
            stderr=b"",
        )

    monkeypatch.setattr(pilot_reader.subprocess, "run", git_show)
    aggregate_sha256 = _sha(aggregate_path)
    freeze = pilot_helpers._freeze_config(
        config,
        beta=written["selection"]["recommended_pilot_freeze_beta"],
        source_sha256=aggregate_sha256,
    )
    freeze["design"]["name"] = "common-beta-post-recovery-freeze-test"
    binding = pilot_reader.verify_beta_source_aggregate(freeze, aggregate_path)
    assert binding is not None
    assert binding["sha256"] == aggregate_sha256

    mirror_root = project / "arbitrary-mirror"
    mirror_root.mkdir()
    mirror = mirror_root / aggregate_path.name
    mirror.write_bytes(aggregate_path.read_bytes())
    mirror_freeze = pilot_helpers._freeze_config(
        config,
        beta=written["selection"]["recommended_pilot_freeze_beta"],
        source_sha256=_sha(mirror),
    )
    mirror_freeze["design"]["name"] = "common-beta-post-recovery-mirror-test"
    with pytest.raises(ValueError, match="locked namespace"):
        pilot_reader.verify_beta_source_aggregate(mirror_freeze, mirror)

    alias_name = "common_beta_post_recovery_calibration_alias.yaml"
    alias_overlay = evidence_configs / alias_name
    alias_overlay.write_bytes(overlay_raw)
    git_objects[f"{'a' * 40}:configs/{alias_name}"] = overlay_raw
    forged_alias = copy.deepcopy(written)
    forged_alias["post_recovery_control"]["phase2_overlay"] = str(
        forged_alias["post_recovery_control"]["phase2_overlay"]
    ).replace("common_beta_post_recovery_calibration.yaml", alias_name)
    forged_alias["post_recovery_control"]["phase2_overlay_repo_relative"] = f"configs/{alias_name}"
    _write_json(aggregate_path, forged_alias)
    alias_sha256 = _sha(aggregate_path)
    alias_freeze = pilot_helpers._freeze_config(
        config,
        beta=written["selection"]["recommended_pilot_freeze_beta"],
        source_sha256=alias_sha256,
    )
    alias_freeze["design"]["name"] = "common-beta-post-recovery-overlay-alias-test"
    with pytest.raises(ValueError, match="exact semantic name"):
        pilot_reader.verify_beta_source_aggregate(alias_freeze, aggregate_path)
    _write_json(aggregate_path, written)
    aggregate_sha256 = _sha(aggregate_path)

    # Confirmatory consumes an accepted freeze v3 through the same recursive
    # reader; it never publishes a pilot-v3 aggregate of its own.
    frozen_beta = float(written["selection"]["recommended_pilot_freeze_beta"])
    freeze_config = pilot_helpers._freeze_config(
        config,
        beta=frozen_beta,
        source_sha256=aggregate_sha256,
    )
    freeze_config["design"]["name"] = "common-beta-post-recovery-freeze-consumer-test"
    freeze_root = project / "runs" / "phase2-post-recovery-freeze"
    freeze_root.mkdir(parents=True)
    freeze_results = [
        pilot_helpers._seed_result(
            freeze_root,
            freeze_config,
            seed=seed,
            beta=frozen_beta,
        )
        for seed in (20260801, 20260802, 20260803)
    ]
    for task, (seed, result) in enumerate(
        zip((20260801, 20260802, 20260803), freeze_results, strict=True)
    ):
        _upgrade_run(
            result.parent,
            helpers=aggregate_helpers,
            config=freeze_config,
            seed=seed,
            task=task,
            authorization_sha256=authorization_sha256,
        )
    freeze_design_sha256 = phase2_design_identity(freeze_config)
    monkeypatch.setattr(subprocess, "run", real_subprocess_run)
    for task, (seed, result) in enumerate(
        zip((20260801, 20260802, 20260803), freeze_results, strict=True)
    ):
        artifact_link = result.parent / "artifact"
        artifact_target = (
            project
            / "artifacts"
            / "phase2-post-recovery-freeze"
            / freeze_design_sha256
            / f"seed-{seed}"
            / f"job-1000_{task}"
        )
        artifact_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(artifact_link, artifact_target)
        shutil.rmtree(artifact_link)
        _link_directory(artifact_link, artifact_target)
    freeze_terminal = freeze_root / "terminal-1000.json"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=raw_sacct,
            stderr=b"",
        ),
    )
    control.capture_post_recovery_terminal_evidence(
        "1000",
        freeze_terminal,
        pilot_phase="freeze",
    )
    monkeypatch.setattr(subprocess, "run", git_show)
    freeze_aggregate_path = output_root / "phase2-post-recovery-freeze-aggregate.json"
    freeze_evidence_root = Path(f"{freeze_aggregate_path}.evidence")
    freeze_evidence_configs = freeze_evidence_root / "configs"
    freeze_evidence_configs.mkdir(parents=True)
    freeze_overlay = freeze_evidence_configs / "common_beta_post_recovery_freeze.yaml"
    freeze_overlay.write_text(
        yaml.safe_dump(freeze_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    (freeze_evidence_configs / "common_beta_pilot_base.yaml").write_bytes(
        (ROOT / "configs" / "common_beta_pilot_base.yaml").read_bytes()
    )
    freeze_overlay_raw = freeze_overlay.read_bytes()
    git_objects[f"{'a' * 40}:configs/common_beta_post_recovery_freeze.yaml"] = freeze_overlay_raw
    freeze_written = aggregate.write_phase2_post_recovery_aggregate(
        freeze_overlay,
        freeze_results,
        freeze_aggregate_path,
        authorization_path=authorization,
        authorization_sha256=authorization_sha256,
        terminal_evidence_path=freeze_terminal,
        terminal_evidence_sha256=_sha(freeze_terminal),
        array_job_id="1000",
        submission_intent_sha256=PILOT_SUBMISSION_INTENT_SHA256,
        submission_ledger_sha256=PILOT_SUBMISSION_LEDGER_SHA256,
        submission_intent_reference_path=(
            freeze_evidence_root / "submission-registry" / "intent.json"
        ),
        submission_ledger_reference_path=(
            freeze_evidence_root / "submission-registry" / "submission.json"
        ),
        aggregator_git_commit="f" * 40,
        producer_git_commit="a" * 40,
        image_sha256="b" * 64,
        hf_inventory_sha256="c" * 64,
        reference_base=output_root,
        phase2_overlay_reference_path=freeze_overlay,
        beta_source_aggregate_path=aggregate_path,
        horizon_parent_aggregate_path=aggregate_path,
    )
    assert freeze_written["selection"]["accepted_for_confirmatory_identity"] is True
    freeze_aggregate_sha256 = _sha(freeze_aggregate_path)
    config_helpers = _config_helpers()
    confirmatory_overlay, confirmatory_base = config_helpers._future_confirmatory(freeze_config)
    confirmatory_overlay["design"]["name"] = "common-beta-post-recovery-confirmatory-consumer-test"
    confirmatory_overlay["objective"]["common_beta"]["frozen_global_beta"] = frozen_beta
    confirmatory_overlay["objective"]["common_beta"]["beta_source_aggregate_sha256"] = (
        freeze_aggregate_sha256
    )
    confirmatory_overlay["evaluation"]["max_length"]["parent_pilot_aggregate_sha256"] = (
        freeze_aggregate_sha256
    )
    confirmatory_config = validate_phase2_config(
        confirmatory_overlay,
        base_config=confirmatory_base,
    )
    confirmatory_beta_binding = pilot_reader.verify_beta_source_aggregate(
        confirmatory_config,
        freeze_aggregate_path,
    )
    confirmatory_horizon_binding = pilot_reader.verify_horizon_parent_aggregate(
        confirmatory_config,
        freeze_aggregate_path,
    )
    assert confirmatory_beta_binding is not None
    assert confirmatory_beta_binding["accepted_beta"] == frozen_beta
    assert confirmatory_horizon_binding is not None
    assert confirmatory_horizon_binding["source_pilot_phase"] == "freeze"
    assert confirmatory_config["design"]["stage"] == "confirmatory"
    assert confirmatory_config["schema_version"] != ("common-beta-pilot-selection-aggregate/v3")

    # Rehashing a forged scientific decision must not make the v3 aggregate
    # authoritative; selection is deterministically rebuilt from seed bytes.
    forged_selection = copy.deepcopy(written)
    forged_selection["selection"]["recommended_pilot_freeze_beta"] += 0.25
    _write_json(aggregate_path, forged_selection)
    forged_selection_sha256 = _sha(aggregate_path)
    forged_selection_freeze = pilot_helpers._freeze_config(
        config,
        beta=forged_selection["selection"]["recommended_pilot_freeze_beta"],
        source_sha256=forged_selection_sha256,
    )
    forged_selection_freeze["design"]["name"] = (
        "common-beta-post-recovery-freeze-selection-forgery-test"
    )
    with pytest.raises(ValueError, match="selection differs from recomputed"):
        pilot_reader.verify_beta_source_aggregate(
            forged_selection_freeze,
            aggregate_path,
        )
    _write_json(aggregate_path, written)

    # The runtime receipt is not an independent source of truth: it must equal
    # the runtime compiled from the immutable, Git-bound overlay.
    forged_runtime = copy.deepcopy(written)
    forged_runtime["phase2_runtime_contract"]["relative_damping"] *= 2.0
    forged_runtime["phase2_runtime_contract_sha256"] = aggregate_helpers._canonical_sha256(
        forged_runtime["phase2_runtime_contract"]
    )
    _write_json(aggregate_path, forged_runtime)
    forged_runtime_sha256 = _sha(aggregate_path)
    forged_runtime_freeze = pilot_helpers._freeze_config(
        config,
        beta=written["selection"]["recommended_pilot_freeze_beta"],
        source_sha256=forged_runtime_sha256,
    )
    forged_runtime_freeze["design"]["name"] = (
        "common-beta-post-recovery-freeze-runtime-forgery-test"
    )
    with pytest.raises(ValueError, match="embedded Phase-2 config breaks"):
        pilot_reader.verify_beta_source_aggregate(
            forged_runtime_freeze,
            aggregate_path,
        )
    _write_json(aggregate_path, written)

    # Every scientific decision field is derived data. Rebinding the outer
    # aggregate SHA cannot make a forged selection, seed summary, horizon,
    # geometry, threshold, or information-boundary statement authoritative.
    top_level_forgeries = (
        ("selection", ("selection", "next_action"), "forged_action"),
        (
            "per-seed",
            ("per_seed", "20260801", "pre_oracle_safety_passed"),
            (not written["per_seed"]["20260801"]["pre_oracle_safety_passed"]),
        ),
        (
            "horizon",
            ("horizon", "candidate_horizon_tokens"),
            written["horizon"]["candidate_horizon_tokens"] + 1,
        ),
        (
            "geometry",
            ("rollout_geometry", "materialized_prompts"),
            written["rollout_geometry"]["materialized_prompts"] + 1,
        ),
        (
            "threshold",
            ("thresholds", "mean_policy_to_reference_kl_cap"),
            0.021,
        ),
        (
            "information-boundary",
            ("information_boundary", "heldout_evaluator_called"),
            True,
        ),
    )
    for case, keys, forged_value in top_level_forgeries:
        forged_top_level = copy.deepcopy(written)
        cursor = forged_top_level
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = forged_value
        _write_json(aggregate_path, forged_top_level)
        forged_top_level_sha256 = _sha(aggregate_path)
        forged_top_level_freeze = pilot_helpers._freeze_config(
            config,
            beta=written["selection"]["recommended_pilot_freeze_beta"],
            source_sha256=forged_top_level_sha256,
        )
        forged_top_level_freeze["design"]["name"] = (
            f"common-beta-post-recovery-freeze-{case}-forgery-test"
        )
        with pytest.raises(ValueError):
            pilot_reader.verify_beta_source_aggregate(
                forged_top_level_freeze,
                aggregate_path,
            )
    _write_json(aggregate_path, written)

    # Change a decay-stage learning rate, then recompute every enclosing byte
    # hash and receipt. A hash-only reader would accept this coherent forgery;
    # the native five-head validator must reject the schedule arithmetic.
    first_result = results[0]
    result_value = json.loads(first_result.read_text(encoding="utf-8"))
    result_value["head_training"]["audit"]["primary_heads"]["bt_mle"]["first_order_convergence"][
        "checks"
    ][-1]["learning_rate_used_for_update"] = 9.0e-4
    _write_json(first_result, result_value)
    result_sha256 = _sha(first_result)

    run = first_result.parent
    strict_path = run / "post-recovery-output-verification.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict["result_sha256"] = result_sha256
    _write_json(strict_path, strict)
    strict_sha256 = _sha(strict_path)

    marker_path = run / "SUCCESS"
    marker: dict[str, str] = {}
    for line in marker_path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        marker[key] = value
    marker["phase2_result_sha256"] = result_sha256
    marker["post_recovery_output_verification_sha256"] = strict_sha256
    marker_path.write_text(
        "".join(f"{key}={value}\n" for key, value in marker.items()),
        encoding="utf-8",
        newline="\n",
    )

    forged = json.loads(aggregate_path.read_text(encoding="utf-8"))
    forged_source = forged["sources"][0]
    forged_source["result_sha256"] = result_sha256
    forged_source["post_recovery_output_verification_sha256"] = strict_sha256
    forged_source["success_receipt_sha256"] = _sha(marker_path)
    _write_json(aggregate_path, forged)
    forged_sha256 = _sha(aggregate_path)
    forged_freeze = pilot_helpers._freeze_config(
        config,
        beta=written["selection"]["recommended_pilot_freeze_beta"],
        source_sha256=forged_sha256,
    )
    forged_freeze["design"]["name"] = "common-beta-post-recovery-freeze-test"
    with pytest.raises(ValueError, match="learning rate is not schedule-bound"):
        pilot_reader.verify_beta_source_aggregate(
            forged_freeze,
            aggregate_path,
        )
