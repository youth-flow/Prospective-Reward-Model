from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_calibration.sh"
GENERIC_SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_pilot.sh"
JOB = ROOT / "scripts" / "hpc4" / "phase2_post_recovery_calibration.sbatch"
AGGREGATE_SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_aggregate.sh"
AGGREGATE_JOB = ROOT / "scripts" / "hpc4" / "phase2_post_recovery_aggregate.sbatch"
SUBMIT_ONCE = ROOT / "scripts" / "hpc4" / "submit_phase2_post_recovery_array_once.py"


def test_submission_is_one_fixed_three_seed_l20_array() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    generic = GENERIC_SUBMIT.read_text(encoding="utf-8")
    helper = SUBMIT_ONCE.read_text(encoding="utf-8")
    assert "submit_phase2_post_recovery_array_once.py" in generic
    assert "--clusters=hpc4" in helper
    assert "--account=sigroup" in helper
    assert "--partition=gpu-l20" in helper
    assert "--gpus-per-node=1" in helper
    assert 'ARRAY_SPEC = "0-2%2"' in helper
    assert "--no-requeue" in helper
    assert "PRORM_PHASE2_ARRAY_CONCURRENCY" not in generic
    assert "array_selection" not in generic
    assert "configs/common_beta_post_recovery_calibration.yaml" in submit
    assert 'configs/common_beta_pilot.yaml"' not in generic
    assert "fixed combined Gate-R/Gate-C authorization" in submit
    assert "exactly one immutable three-seed array" in submit
    assert "--legacy-r2-replay" not in submit
    assert 'authorization_mode="active-r3"' in generic
    assert '--authorization-mode "${authorization_mode}"' in generic


def test_job_locks_scheduler_and_uses_only_fresh_execution() -> None:
    job = JOB.read_text(encoding="utf-8")
    for fragment in (
        "#SBATCH --partition=gpu-l20",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --array=0-2%2",
        "#SBATCH --no-requeue",
        '[[ "${SLURM_CLUSTER_NAME:-}" = "hpc4" ]]',
        '[[ "${SLURM_RESTART_COUNT:-0}" = "0" ]]',
        '[[ "${SLURM_ARRAY_TASK_COUNT:-}" = "3" ]]',
        '[[ "${SLURM_ARRAY_TASK_MIN:-}" = "0" ]]',
        '[[ "${SLURM_ARRAY_TASK_MAX:-}" = "2" ]]',
        '[[ "${SLURM_ARRAY_TASK_STEP:-}" = "1" ]]',
        "seeds=(20260801 20260802 20260803)",
    ):
        assert fragment in job
    assert job.count("controlled-materialize") == 2  # comment plus command
    assert "prepare_phase2_inputs" in job
    assert "phase2-run" in job
    materialize_call = "python -m smart_reward.cli controlled-materialize"
    prepare_call = "inputs=prepare_phase2_inputs("
    phase2_call = "python -m smart_reward.cli phase2-run"
    assert job.index(materialize_call) < job.index(prepare_call)
    assert job.index(prepare_call) < job.index(phase2_call)
    assert "materialization_mode=fresh" in job
    assert "recovery_outputs_mounted=false" in job
    assert "schema_version=prorm-phase2-post-recovery-pilot-run-status/v1" in job
    assert "pilot_phase=%s" in job
    assert "PRORM_POST_RECOVERY_PILOT_PHASE" in job
    assert "slurm_job_id=%s" in job
    assert "allocation_job_id_raw=%s" in job
    assert "slurm_array_task_job_id=%s" in job
    assert '--slurm-job-id-raw "${SLURM_JOB_ID}"' in job
    assert '--pilot-phase "${PRORM_POST_RECOVERY_PILOT_PHASE}"' in job
    assert "post-recovery-calibration-run-status/v1" not in job
    assert 'artifact_mode="reused"' not in job
    assert 'cp -a -- "${project_artifact}"' not in job


def test_container_mounts_only_job_writable_and_immutable_inputs_read_only() -> None:
    job = JOB.read_text(encoding="utf-8")
    assert '--bind "${job_dir}:${job_dir}"' in job
    assert '--bind "${PRORM_HF_CACHE}:${PRORM_HF_CACHE}:ro"' in job
    assert ('--bind "${PRORM_RECOVERY_AUTHORIZATION}:${PRORM_RECOVERY_AUTHORIZATION}:ro"') in job
    assert "authorization_dependency_binds+=(" in job
    assert '--bind "${dependency_path}:${dependency_path}:ro"' in job
    assert '--env "HF_DATASETS_CACHE=${datasets_cache}"' in job
    assert '--env "TMPDIR=${tmp_dir}"' in job
    assert '--env "XDG_CACHE_HOME=${xdg_cache}"' in job
    assert '--verification-datasets-cache "${datasets_cache}"' in job
    assert "${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}" not in job
    assert "/phase2-recovery-pilot/" not in job
    assert "/parent-artifact" not in job


def test_job_revalidates_auth_and_detached_commit_before_and_after() -> None:
    job = JOB.read_text(encoding="utf-8")
    assert "git clone --quiet --no-hardlinks --no-checkout" in job
    assert 'checkout --quiet --detach "${PRORM_GIT_COMMIT}"' in job
    assert job.count("validate_phase2_recovery_authorization.py") == 2
    assert job.count("inspect_phase2_post_recovery_stdlib.py") == 1
    assert "authorization-check-host-before.json" in job
    assert "authorization-check-container.json" in job
    assert "authorization-check-host-after.json" in job
    assert "authorization-check-container-after.json" in job
    assert "post-recovery-output-verification.json" in job
    assert "optimizer_schedule_sha256=%s" in job
    assert "PRORM_POST_RECOVERY_AUTHORIZATION_MODE" in job
    assert '--authorization-mode "${PRORM_POST_RECOVERY_AUTHORIZATION_MODE}"' in job
    assert '"${authorization_validation_options[@]}"' in job


def test_aggregate_is_gated_by_terminal_evidence_and_has_no_project_rw_bind() -> None:
    submit = AGGREGATE_SUBMIT.read_text(encoding="utf-8")
    job = AGGREGATE_JOB.read_text(encoding="utf-8")
    assert 'capture_phase2_post_recovery_terminal.py" verify' in submit
    assert '--pilot-phase "${pilot_phase}"' in submit
    assert "inspect_phase2_post_recovery_stdlib.py" in submit
    assert "terminal evidence path differs from the semantic pilot phase" in submit
    assert "post-recovery aggregate output must use its locked semantic path" in submit
    assert 'semantic_output_name="${identities[6]}"' in submit
    assert "PRORM_POST_RECOVERY_TERMINAL_SHA256" in submit
    assert "PRORM_POST_RECOVERY_ARRAY_JOB_ID" in submit
    assert "run ${task} SUCCESS marker" in submit
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --partition=gpu-l20" in job
    assert "#SBATCH --gpus-per-node=1" in job
    assert '[[ "${SLURM_RESTART_COUNT:-0}" = "0" ]]' in job
    assert "--authorization-sha256" in job
    assert "--terminal-evidence-sha256" in job
    assert "--array-job-id" in job
    assert "--reference-base" in job
    assert "--phase2-overlay-reference" in job
    assert "phase2_overlay_git_blob_sha1" in job
    assert '"${output}.evidence"' in job
    assert "${PRORM_PHASE2_BASE_REL}" in job
    assert "${artifacts[$task]}:${artifacts[$task]}:ro" in job
    assert "authorization-check-host.json" in job
    assert 'binds+=",${dependency_path}:${dependency_path}:ro"' in job
    assert ":ro" in job
    assert "${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}" not in job
    assert "common-beta-pilot-selection-aggregate/v3" in job
    assert "all_tasks_terminal_completed_zero_exit" in job
    assert 'control.get("pilot_phase")!=sys.argv[9]' in job
    assert 'control.get("pilot_terminal_evidence_sha256")' in job
    assert 'control.get("pilot_array_job_id")' in job
    assert "PRORM_POST_RECOVERY_AUTHORIZATION_MODE" in submit
    assert "PRORM_POST_RECOVERY_AUTHORIZATION_MODE" in job
    assert '--authorization-mode "${PRORM_POST_RECOVERY_AUTHORIZATION_MODE}"' in job
    assert '"${authorization_validation_options[@]}"' in job


def test_old_replay_entrypoints_preserve_v2_and_route_only_post_recovery_schema() -> None:
    old_submit = (ROOT / "scripts" / "hpc4" / "submit_phase2_pilot.sh").read_text(encoding="utf-8")
    old_job = (ROOT / "scripts" / "hpc4" / "phase2_pilot.sbatch").read_text(encoding="utf-8")
    assert "prorm-common-beta-post-recovery-experiment/v1" in old_submit
    assert "submit_phase2_post_recovery_pilot.sh" in old_submit
    assert "--legacy-r2-replay" in old_submit
    assert '[[ "${overlay_relative}" = "configs/common_beta_pilot.yaml" ]]' in old_submit
    assert "prorm-common-beta-config/v2" in old_submit
    assert "post_recovery" not in old_job
    assert "common_beta_pilot.yaml" in old_submit
