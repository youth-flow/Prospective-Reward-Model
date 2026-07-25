from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_confirmatory.sh"
JOB = ROOT / "scripts" / "hpc4" / "phase2_confirmatory.sbatch"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _bash() -> str | None:
    discovered = shutil.which("bash")
    if discovered is not None:
        return discovered
    for candidate in (
        Path("D:/Git/bin/bash.exe"),
        Path("C:/Program Files/Git/bin/bash.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _shell_function_source(source: str, name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


def _embedded_python_sources(source: str) -> list[str]:
    marker = "<<'PY'\n"
    programs: list[str] = []
    cursor = 0
    while True:
        start = source.find(marker, cursor)
        if start < 0:
            return programs
        start += len(marker)
        end = source.find("\nPY\n", start)
        assert end >= 0, "unterminated embedded Python heredoc"
        programs.append(source[start:end])
        cursor = end + len("\nPY\n")


def test_confirmatory_sources_and_embedded_python_are_parseable() -> None:
    for path in (SUBMIT, JOB):
        source = _text(path)
        programs = _embedded_python_sources(source)
        assert programs
        for program in programs:
            compile(program, f"{path.name}:heredoc", "exec")

    bash = _bash()
    if bash is None:
        return
    for path in (SUBMIT, JOB):
        result = subprocess.run(
            [bash, "-n", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_submit_requires_frozen_formal_identity_and_exact_seed_order() -> None:
    submit = _text(SUBMIT)

    assert (
        "<overlay.yaml> <base.yaml> <accepted-freeze-aggregate.json> "
        "<gpu-partition> <walltime> [0-29]"
    ) in submit
    assert "Phase-2 confirmatory submission requires a clean Git worktree" in submit
    assert "configs/identities.json" in submit
    assert '"cat-file", "blob"' in submit
    assert "worktree overlay bytes do not match submitted Git commit" in submit
    assert "committed config bytes do not match committed identity" in submit
    assert "the Phase-2 confirmatory identity must declare exactly 30 seeds" in submit
    assert "confirmatory run.seeds must be exactly ordered 20260901 through 20260930" in submit
    assert "expected_seeds = list(range(20260901, 20260931))" in submit
    assert "stage:[ \\t]*confirmatory" in submit
    assert "formal_eligibility:[ \\t]*true" in submit
    assert "excluded_from_confirmatory_evidence:[ \\t]*false" in submit
    assert "confirmatory submission refuses the pilot overlay" in submit
    assert "confirmatory submission refuses the pilot base config" in submit


def test_submit_requires_one_accepted_freeze_for_beta_and_horizon() -> None:
    submit = _text(SUBMIT)

    assert "beta_source_aggregate_sha256" in submit
    assert "parent_pilot_aggregate_sha256" in submit
    assert "confirmatory beta and horizon must bind the same accepted freeze aggregate" in submit
    assert "accepted freeze aggregate bytes do not match the confirmatory design binding" in submit
    assert "common-beta-pilot-selection-aggregate/v2" in submit
    assert "phase2-pilot-aggregation-identity/v1" in submit
    assert "pilot-freeze-selection/v1" in submit
    for required_gate in (
        "selection_accepted",
        "accepted_for_confirmatory_identity",
        "all_seeds_and_arms_used_same_beta",
        "all_pre_oracle_safety_gates_passed",
        "all_length_gates_passed",
        "all_non_length_safety_gates_passed",
    ):
        assert required_gate in submit
    assert "freeze_confirmatory_design_identity" in submit
    assert "accepted freeze beta differs from confirmatory frozen_global_beta" in submit
    assert "PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256=" in submit
    assert "PRORM_PHASE2_FROZEN_GLOBAL_BETA=" in submit


def test_submit_accepts_only_the_complete_exact_30_array() -> None:
    submit = _text(SUBMIT)
    assert "formal campaign must submit the exact complete seed array" in submit
    assert (
        "initial campaign registry commit must reserve exact ordered tasks 0 through 29" in submit
    )
    assert 'or [entry["array_task_id"] for entry in entries] != list(range(30))' in submit
    assert 'array_spec="${array_start}-${array_end}%${concurrency}"' in submit
    assert "PRORM_PHASE2_ARRAY_CONCURRENCY must be 1 or 2" in submit
    assert 'value.get("array_spec") not in {"0-29%1", "0-29%2"}' in submit
    assert ("^[1-9][0-9]*-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$") in submit

    bash = _bash()
    if bash is None:
        return
    predicate = _shell_function_source(submit, "formal_array_shape_is_valid")
    for start, end, count, accepted in (
        (0, 29, 30, True),
        (0, 5, 30, False),
        (7, 7, 30, False),
        (1, 29, 30, False),
        (0, 30, 31, False),
    ):
        result = subprocess.run(
            [
                bash,
                "-c",
                f"{predicate}\nformal_array_shape_is_valid {start} {end} {count}",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert (result.returncode == 0) is accepted


def test_submit_is_l20_locked_held_no_requeue_and_cluster_exact() -> None:
    submit = _text(SUBMIT)

    assert "apptainer exec" not in submit
    assert "gpu-l20) ;;" in submit
    for forbidden_partition in ("gpu-a30", "gpu-rtx5880", "gpu-rtx4090d"):
        assert forbidden_partition not in submit
    assert "submission forbids ambient sbatch option overrides" in submit
    assert "submission forbids ambient container controls" in submit
    assert '--export="ALL,' not in submit
    assert "--account=sigroup" in submit
    assert "--gpus-per-node=1" in submit
    assert "--cpus-per-task=8" in submit
    assert "--mem=64G" in submit
    assert "--hold" in submit
    assert "--no-requeue" in submit
    assert "--signal=B:USR1@120" in submit
    assert 'awk \'$1 == "ClusterName" && $2 == "=" {print $3}\'' in submit
    assert '[[ "${configured_cluster}" = "hpc4" ]]' in submit
    assert 'submitted_cluster="${configured_cluster}"' in submit
    assert "sbatch cluster identity differs from scontrol ClusterName" in submit


def test_registry_commit_is_durable_before_release_and_fail_closed() -> None:
    submit = _text(SUBMIT)

    assert 'campaign_registry="${formal_design_root}/campaign-registry"' in submit
    assert 'exec {registry_lock_fd}> "${registry_lock}"' in submit
    assert "formal campaign registry lock is not a non-symlink regular file" in submit
    assert "prorm-phase2-campaign-submission/v1" in submit
    assert "committed_while_slurm_held" in submit
    assert "mv -T --no-clobber --" in submit
    held_submit = submit.index("submission_output=")
    registry_commit = submit.index(
        'mv -T --no-clobber -- "${submission_staging}" "${submission_record}"'
    )
    record_fsync = submit.index(
        'python3 -I -S - "${submission_record}"',
        registry_commit,
    )
    release = submit.index('scontrol release "${held_array_job_id}"')
    assert held_submit < registry_commit < record_fsync < release
    cleanup = submit[submit.index("cleanup_held_array()") : held_submit]
    assert 'scancel -- "${held_array_job_id}"' in cleanup
    assert "submission_record" not in cleanup
    assert "rm " not in cleanup
    assert "for directory in (path, path.parent):" in submit
    lock = submit.index('flock -x "${registry_lock_fd}"')
    no_recovery = submit.index(
        "formal no-retry campaign recovery registry must remain empty",
        lock,
    )
    resume = submit.index("committed_submission_info", no_recovery)
    assert lock < no_recovery < resume


def test_commit_release_crash_resumes_exact_registered_array_without_sbatch() -> None:
    submit = _text(SUBMIT)

    resume_comment = submit.index(
        "A SIGKILL after the immutable registry commit but before `scontrol release`"
    )
    committed_scan = submit.index("committed_submission_info", resume_comment)
    scheduler_lookup = submit.index(
        'scontrol show job --oneliner "${committed_array_job_id}"',
        committed_scan,
    )
    resumed_release = submit.index(
        'scontrol release "${committed_array_job_id}"',
        scheduler_lookup,
    )
    resume_exit = submit.index("exit 0", resumed_release)
    fresh_submit = submit.index("submission_output=")
    assert resume_comment < committed_scan < scheduler_lookup < resumed_release
    assert resumed_release < resume_exit < fresh_submit
    resume_block = submit[resume_comment:fresh_submit]
    assert "committed campaign submission differs from this exact invocation" in resume_block
    assert "Slurm job differs from the committed held-array identity" in resume_block
    assert 'parsed[0].get("ArrayTaskId") == array_spec' in resume_block
    assert 'parsed[0].get("Reason") == "JobHeldUser"' in resume_block
    assert "records = [line for line in record.splitlines() if line]" in resume_block
    assert "committed array is held by an unexpected scheduler authority" in resume_block
    assert "ALREADY_RELEASED" in resume_block
    assert "sbatch " not in resume_block
    assert "printf '%s;%s\\n'" in resume_block


def test_formal_contract_has_one_predeclared_attempt_and_no_retry_surface() -> None:
    submit = _text(SUBMIT)
    job = _text(JOB)

    assert "--prior-attempt-ledger" not in submit
    assert "PRORM_PHASE2_PRIOR" not in submit
    assert "PRORM_PHASE2_PRIOR" not in job
    assert "publish_pre_outcome_recovery" not in job
    assert "retry_eligible" not in job
    assert '[[ "${PRORM_PHASE2_ATTEMPT_INDEX}" = "1" ]]' in job
    assert "phase2-seed-attempt-ledger/v3" in job
    assert '"retry_policy": "single_predeclared_attempt_no_retry"' in job
    assert "phase2-seed-attempt-ledger/v2" not in job
    assert '"replacement_seed_allowed": False' in job


def test_job_is_detached_offline_and_rechecks_formal_compute_identity() -> None:
    job = _text(JOB)

    assert "#SBATCH --job-name=prorm-phase2-confirmatory" in job
    assert "#SBATCH --account=sigroup" in job
    assert "#SBATCH --gpus-per-node=1" in job
    assert '[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]]' in job
    assert '[[ "${SLURM_CLUSTER_NAME}" = "hpc4" ]]' in job
    assert "gpu-l20) ;;" in job
    assert "exactly one Slurm GPU" in job
    assert "CUDA_VISIBLE_DEVICES must identify exactly one allocated GPU" in job
    assert "torch.cuda.device_count() != 1" in job
    assert 'partition != "gpu-l20" or "l20" not in name.lower()' in job
    assert "apptainer exec --cleanenv --nv" in job
    assert "--no-mount home,cwd,bind-paths" in job
    project_evidence_bind = '--bind "${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}:ro"'
    cache_bind = '--bind "${job_dir}:${job_dir},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}"'
    assert project_evidence_bind in job
    assert job.index(project_evidence_bind) < job.index(cache_bind)
    assert (
        '--bind "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}:'
        '${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}"'
    ) not in job
    assert "recursively revalidate those bytes" in job
    assert '--env "HF_HUB_OFFLINE=1"' in job
    assert '--env "TRANSFORMERS_OFFLINE=1"' in job
    assert '--env "HF_DATASETS_OFFLINE=1"' in job
    assert "--allow-download" not in job
    assert "git clone --quiet --no-hardlinks --no-checkout" in job
    assert 'checkout --quiet --detach "${PRORM_GIT_COMMIT}"' in job
    assert "detached execution checkout became dirty" in job
    assert "formal Phase-2 compute must execute on Slurm cluster hpc4" in job
    assert "array task does not preserve the preregistered formal seed order" in job
    assert "formal compute identity differs from the exact held-array schema" in job
    assert "set(value) != expected_fields" in job
    assert 'value.get("entries") != expected_entries' in job
    assert 're.fullmatch(r"0-29%[12]"' in job
    assert 'set(job_tuple) != set(expected_job_tuple) | {"walltime"}' in job
    assert 'value.get("producer") != expected_producer' in job


def test_job_rejects_requeue_before_any_compute_or_attempt_claim() -> None:
    job = _text(JOB)

    restart_rejection = job.index(
        "automatic Slurm requeue/restart is forbidden for a formal Phase-2 attempt"
    )
    execution_claim = job.index('registry_execution="${registry_executions}/seed-${seed}-attempt-')
    attempt_claim = job.index('attempt_root="${seed_root}/attempt-${PRORM_PHASE2_ATTEMPT_INDEX}"')
    assert restart_rejection < execution_claim
    assert restart_rejection < attempt_claim
    assert 'case "${SLURM_RESTART_COUNT:-0}" in' in job


def test_job_claims_once_and_uses_hidden_transactional_job_staging() -> None:
    job = _text(JOB)

    assert "prorm-phase2-campaign-execution/v1" in job
    assert "compute_started_no_requeue" in job
    assert "formal seed attempt already has a registered compute execution" in job
    assert 'mv -T --no-clobber -- "${registry_execution_staging}"' in job
    assert 'mkdir -- "${attempt_root}"' in job
    assert 'mkdir -p "${attempt_root}"' not in job
    assert "A hard termination after this point deliberately leaves the claim occupied." in job
    assert "prorm-phase2-formal-attempt-claim/v1" in job
    assert 'project_run_final="${attempt_root}/job-${execution_id}"' in job
    assert 'project_run="${attempt_root}/.job-${execution_id}.in-progress-${SLURM_JOB_ID}"' in job
    execution_publish = job.index('mv -T --no-clobber -- "${registry_execution_staging}"')
    admission_lock = job.index('flock -x "${execution_registry_lock_fd}"')
    no_recovery = job.index(
        "formal no-retry campaign recovery registry must remain empty",
        admission_lock,
    )
    admission_unlock = job.index('flock -u "${execution_registry_lock_fd}"')
    directory_claim = job.index('mkdir -- "${attempt_root}"')
    claim_publish = job.index(
        'mv -T --no-clobber -- "${attempt_claim_temporary}" "${attempt_claim}"'
    )
    staging_create = job.index('mkdir -- "${project_run}"')
    workload = job.index("python -m smart_reward.cli phase2-run")
    assert admission_lock < no_recovery < execution_publish < admission_unlock
    assert admission_unlock < directory_claim < claim_publish < staging_create < workload


def test_outcome_boundary_precedes_every_formal_artifact_operation() -> None:
    job = _text(JOB)

    marker_path = 'outcome_reveal_marker="${attempt_root}/OUTCOME_REVEAL_STARTED"'
    marker_publish = (
        'mv -T --no-clobber -- \\\n  "${outcome_reveal_temporary}" "${outcome_reveal_marker}"'
    )
    assert marker_path in job
    assert "prorm-phase2-outcome-reveal-boundary/v1" in job
    durability = job.index('fsync_file_and_parent "${outcome_reveal_marker}"')
    artifact_root = job.index(
        'ensure_shared_directory "project artifacts root"',
    )
    artifact_address = job.index(
        'project_artifact="${artifact_parent}/seed-${seed}"',
    )
    materialize = job.index("python -m smart_reward.cli controlled-materialize")
    outcome_call = job.index("python -m smart_reward.cli phase2-run")
    assert job.index(marker_publish) < durability
    assert durability < artifact_root < artifact_address < materialize < outcome_call
    assert "pre-existing formal-seed artifact has no current-attempt provenance" in job
    assert 'artifact_mode="reused"' not in job
    assert 'artifact_mode="materialized_by_current_attempt"' in job
    timeout_handler = job[
        job.index("on_pre_outcome_timeout()") : job.index("trap on_pre_outcome_timeout USR1")
    ]
    assert "refresh_outcome_reveal_state" in timeout_handler
    assert "exit 75" in timeout_handler


def test_job_materializes_fresh_artifact_and_runs_registered_phase2() -> None:
    job = _text(JOB)

    assert (
        'artifact_config_root="${PRORM_PROJECT_ROOT}/artifacts/${PRORM_PHASE2_BASE_CONFIG_HASH}"'
    ) in job
    assert 'artifact_image_root="${artifact_config_root}/${PRORM_IMAGE_SHA256}"' in job
    assert ('artifact_inventory_root="${artifact_image_root}/${PRORM_HF_INVENTORY_SHA256}"') in job
    assert 'artifact_parent="${artifact_inventory_root}/${PRORM_GIT_COMMIT}"' in job
    assert 'project_artifact="${artifact_parent}/seed-${seed}"' in job
    artifact_copy = job.index('cp -a -- "${artifact_dir}/." "${artifact_staging}/"')
    artifact_sync = job.index('fsync_tree "${artifact_staging}"')
    artifact_publish = job.index(
        'mv -T --no-clobber -- "${artifact_staging}" "${project_artifact}"'
    )
    artifact_parent_sync = job.index('fsync_directory "${artifact_parent}"')
    assert artifact_copy < artifact_sync < artifact_publish < artifact_parent_sync
    assert "controlled-materialize" in job
    assert 'ln -s -- "${artifact_relative}" "${artifact_link}"' in job
    phase2_command = job.index("python -m smart_reward.cli phase2-run")
    explicit_bridge = job.index("inputs = prepare_phase2_inputs(")
    assert explicit_bridge < phase2_command
    assert '"${execution_overlay}" "${artifact_dir}" "${manifest}" "${phase2_result}"' in job
    assert '--beta-source-aggregate "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}"' in job
    assert '--horizon-parent-aggregate "${PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE}"' in job
    assert 'phase2_result="${job_dir}/phase2-result.json"' in job
    assert 'phase2_rollouts="${job_dir}/phase2-result.rollouts.jsonl"' in job
    assert '"PRORM_FAILURE_EVIDENCE=${job_dir}/phase2-failure-evidence.json"' in job
    assert "phase2-failure-evidence.json" in job
    assert "common-beta-finite-policy/v2" in job
    assert "common-beta-frozen-global/v1" in job
    assert "phase2-pre-oracle-safety-gate/v1" in job


def test_success_is_receipt_sealed_and_rederived_by_finalizer() -> None:
    job = _text(JOB)

    success_ledger = job.index('build_attempt_ledger "success_result" "true"')
    success_terminal = job.index("python -m smart_reward.cli phase2-success-manifest")
    receipt_publish = job.index('"${success_receipt_temporary}" "${success_validation_receipt}"')
    assert success_ledger < success_terminal < receipt_publish
    for immutable_check in (
        "published base artifact metadata changed during formal execution",
        "HF inventory changed during formal execution",
        "research image changed during formal execution",
        "accepted freeze aggregate changed during formal execution",
    ):
        assert job.index(immutable_check) < success_ledger
    assert job.rindex("assert_execution_checkout") < success_ledger
    assert "prorm-phase2-success-validation-receipt/v1" in job
    assert "VALIDATED_SUCCESS_READY_FOR_ATOMIC_JOB_PUBLICATION" in job
    finalizer = job[job.index("finalize_run()") : job.index("trap finalize_run EXIT")]
    assert "refresh_success_terminal_state" in finalizer
    assert '"${success_ledger_source}" "${job_dir}/phase2-attempt-ledger.json"' in finalizer
    assert (
        "formal job prepublication validation failed; leaving only non-authoritative staging"
    ) in finalizer


def test_prepublication_failure_cannot_publish_any_canonical_job(
    tmp_path: Path,
) -> None:
    bash = _bash()
    if bash is None:
        return
    job = _text(JOB)
    finalizer = _shell_function_source(job, "finalize_run")
    staging = tmp_path / ".job-test.in-progress-1"
    canonical = tmp_path / "job-test"
    scratch = tmp_path / "scratch"
    staging.mkdir()
    scratch.mkdir()
    (scratch / "phase2-attempt-ledger.json").write_text("{}\n", encoding="utf-8")
    marker_called = tmp_path / "marker-called"
    script = f"""
set -u
{finalizer}
job_dir={shlex.quote(str(scratch))}
project_run={shlex.quote(str(staging))}
project_run_final={shlex.quote(str(canonical))}
attempt_root={shlex.quote(str(tmp_path))}
registry_lock={shlex.quote(str(tmp_path / "registry.lock"))}
registry_scheduler_terminals={shlex.quote(str(tmp_path / "scheduler-terminals"))}
mkdir -p -- "$registry_scheduler_terminals"
touch -- "$registry_lock"
seed=20260901
PRORM_PHASE2_ATTEMPT_INDEX=1
SLURM_ARRAY_JOB_ID=1
SLURM_ARRAY_TASK_ID=0
SLURM_CLUSTER_NAME=hpc4
SLURM_JOB_ID=1
success_terminal_ready=1
success_ledger_source="$job_dir/phase2-attempt-ledger.json"
final_outcome_reveal_started=1
materialized_this_job=1
refresh_outcome_reveal_state() {{ return 0; }}
refresh_success_terminal_state() {{ return 0; }}
record_stage() {{ return 0; }}
build_attempt_ledger() {{ return 1; }}
atomic_sync_file() {{ return 1; }}
atomic_write_marker() {{ touch -- {shlex.quote(str(marker_called))}; return 0; }}
fsync_tree() {{ return 0; }}
fsync_directory() {{ return 0; }}
true
finalize_run
"""
    result = subprocess.run(
        [bash, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not canonical.exists()
    assert not marker_called.exists()
    assert staging.is_dir()


def test_final_job_publication_is_one_locked_atomic_directory_transaction() -> None:
    job = _text(JOB)
    finalizer = job[job.index("finalize_run()") : job.index("trap finalize_run EXIT")]

    validation_gate = finalizer.index("if (( sync_exit != 0 )); then")
    marker = finalizer.index('atomic_write_marker "${marker}"')
    tree_fsync = finalizer.index('fsync_tree "${project_run}"')
    lock = finalizer.index('flock -x "${final_registry_lock_fd}"')
    scheduler_owner = finalizer.index(
        'scheduler_terminal_record="${registry_scheduler_terminals}/seed-'
    )
    publish = finalizer.index('mv -T --no-clobber -- "${project_run}" "${project_run_final}"')
    parent_fsync = finalizer.index('fsync_directory "${attempt_root}"')
    unlock = finalizer.index('flock -u "${final_registry_lock_fd}"')
    assert validation_gate < marker < tree_fsync < lock
    assert lock < scheduler_owner < publish < parent_fsync < unlock
    assert "scheduler reconciliation already owns this formal attempt terminal" in finalizer
    assert "phase2-success-terminal.json" in finalizer
    assert finalizer.index('"phase2-success-terminal.json" || sync_exit=1') < validation_gate
    fsync_tree = job[job.index("fsync_tree()") : job.index("fsync_directory()")]
    assert "unsafe symlink file in final tree" in fsync_tree


def test_attempt_claim_and_created_shared_directories_are_durable() -> None:
    submit = _text(SUBMIT)
    job = _text(JOB)

    assert "for directory in (path, path.parent):" in submit
    ensure = job[job.index("ensure_shared_directory()") : job.index("fsync_file_and_parent()")]
    assert 'fsync_directory "${path}"' in ensure
    assert 'fsync_directory "$(dirname -- "${path}")"' in ensure
    claim = job[
        job.index('mkdir -- "${attempt_root}"') : job.index('attempt_claim="${attempt_root}/CLAIM"')
    ]
    assert 'fsync_directory "${attempt_root}"' in claim
    assert 'fsync_directory "${seed_root}"' in claim


def test_every_atomic_rename_is_no_clobber() -> None:
    for path in (SUBMIT, JOB):
        lines = _text(path).splitlines()
        rename_lines = [line.strip() for line in lines if line.lstrip().startswith("mv ")]
        assert rename_lines
        assert all(line.startswith("mv -T --no-clobber --") for line in rename_lines)
