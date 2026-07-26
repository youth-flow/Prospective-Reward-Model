from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "hpc4" / "submit_phase2_r3_primary.sh"
SUBMIT = ROOT / "scripts" / "hpc4" / "phase2_r3_primary_submission.sbatch"
SBATCH = ROOT / "scripts" / "hpc4" / "phase2_r3_primary.sbatch"


def test_login_launcher_never_executes_the_container() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "apptainer exec" not in text
    assert "phase2_r3_primary_submission.sbatch" in text
    assert "exec srun" in text
    assert "--partition=gpu-l20" in text
    assert "--gpus-per-node=1" in text
    assert 'export_spec="PATH=/usr/bin:/bin"' in text


def test_submitter_derives_all_scheduler_resources_from_the_pinned_plan() -> None:
    text = SUBMIT.read_text(encoding="utf-8")

    assert "prepare_phase2_r3_primary_submission.py" in text
    assert 'python3 "${prepare_cli}" create' in text
    assert 'python3 "${prepare_cli}" inspect' in text
    assert "--operational-bundle-file-sha256" in text
    assert "--profile-terminal-manifest-file-sha256" in text
    assert "--profile-terminal-raw-sacct-sha256" in text
    assert "cp --no-clobber" in text
    assert "cmp --silent" in text
    assert 'input_root="${input_parent}/${commit}"' in text
    assert 'source_config="${input_root}/common_beta_pilot_base.yaml"' in text
    assert "common_beta_recovery_pilot.yaml" not in text
    assert 'parent_registry="${input_root}/phase2_recovery_parent_failures.json"' in text
    assert "retained input copy differs from clean repository bytes" in text
    assert "${repo_root}/configs/phase2_recovery_r3_science.yaml" in text
    assert "/home/yyangjo/Smart-Reward-Model" in text
    assert "/project/sigroup/smart-reward-model" in text
    assert '--array="0-2%${array_concurrency}"' in text
    assert '--signal="B:USR1@${signal_lead_seconds}"' in text
    assert '--account="${account}"' in text
    assert '--partition="${partition}"' in text
    assert "--nodes=1" in text
    assert "--ntasks=1" in text
    assert "--gpus-per-node=1" in text
    assert "--gpus-per-task" not in text
    assert '--cpus-per-task="${cpus_per_task}"' in text
    assert '--mem="${memory_mib}M"' in text
    assert '--time="${slurm_walltime}"' in text
    for canonical_export in (
        'export PRORM_R3_IMAGE="${image}"',
        'export PRORM_R3_REPO_ROOT="${repo_root}"',
        'export PRORM_R3_PROJECT_ROOT="${project_root}"',
        'export PRORM_R3_SCRATCH_ROOT="${scratch_root}"',
        'export PRORM_R3_HF_CACHE="${hf_cache}"',
        'export PRORM_R3_OPERATIONAL_BUNDLE="${operational_bundle}"',
        'export PRORM_R3_PROFILE_INTENT="${profile_intent}"',
        'export PRORM_R3_PROFILE_RUNTIME_RECEIPT="${profile_runtime_receipt}"',
        ('export PRORM_R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY="${profile_terminal_directory}"'),
    ):
        assert canonical_export in text

    create_command = text.split('python3 "${prepare_cli}" create', 1)[1].split(
        'submission_plan_file_sha256="$(',
        1,
    )[0]
    for forbidden in (
        "--walltime-seconds",
        "--memory-bytes",
        "--cpus-per-task",
        "--array-concurrency",
        "--seed",
        "--head",
        "--heldout",
        "--control",
    ):
        assert forbidden not in create_command


def test_sbatch_reopens_plan_and_passes_exact_runner_contract_through_cleanenv() -> None:
    text = SBATCH.read_text(encoding="utf-8")

    assert "#SBATCH" not in text
    assert text.count("--cleanenv") >= 2
    assert "\n  PRORM_R3_SOURCE_TEST_RECEIPT\n" not in text
    assert '--source-test-receipt "${PRORM_R3_SOURCE_TEST_RECEIPT}"' not in text
    assert "runs/phase2-recovery-r3/inputs/${PRORM_R3_GIT_COMMIT}" in text
    assert "common_beta_pilot_base.yaml" in text
    assert "common_beta_recovery_pilot.yaml" not in text
    assert '[[ "${PRORM_R3_SOURCE_CONFIG}" == "${expected_source_config}" ]]' in text
    assert '[[ "${PRORM_R3_PARENT_REGISTRY}" == "${expected_parent_registry}" ]]' in text
    assert '    --source-config "${source_config}" \\' in text
    assert '    --parent-registry "${parent_registry}" \\' in text
    assert '    --source-config "${PRORM_R3_SOURCE_CONFIG}" \\' not in text
    assert '    --parent-registry "${PRORM_R3_PARENT_REGISTRY}" \\' not in text
    assert "retained source config differs from the clean commit" in text
    assert "retained parent registry differs from the clean commit" in text
    assert '[[ "${SLURM_NNODES:-}" == "1" ]]' in text
    assert '[[ "${SLURM_NTASKS:-}" == "1" ]]' in text
    assert 'case "${SLURM_GPUS_ON_NODE:-}" in' in text
    assert 'python3 "${prepare_cli}" inspect' in text
    assert 'python3 "${runner}"' in text
    for variable in (
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
    ):
        assert f'--env "{variable}=${{{variable}}}"' in text
    for option in (
        "--project-root",
        "--science-config",
        "--source-config",
        "--parent-registry",
        "--parent-registry-file-sha256",
        "--gate0-file-sha256",
        "--gate1-file-sha256",
        "--source-test-receipt-file-sha256",
        "--operational-bundle",
        "--operational-bundle-file-sha256",
        "--profile-allocation-intent",
        "--profile-allocation-intent-file-sha256",
        "--profile-runtime-receipt",
        "--profile-runtime-receipt-file-sha256",
        "--profile-terminal-evidence-directory",
        "--profile-terminal-manifest-file-sha256",
        "--profile-terminal-raw-sacct-sha256",
        "--task-root",
        "--runtime-closure",
    ):
        assert option in text
    for forbidden in (
        "--seed",
        "--head",
        "--heldout",
        "--control",
        "--segment-index",
    ):
        assert forbidden not in text


def test_sbatch_forwards_usr1_and_requires_fresh_segment1_paths() -> None:
    text = SBATCH.read_text(encoding="utf-8")

    assert "trap forward_usr1 USR1" in text
    assert 'kill -USR1 "${runner_pid}"' in text
    assert 'task_id="${SLURM_ARRAY_TASK_ID}"' in text
    assert 'task_root="${task_root_base}/task-${task_id}"' in text
    assert "segment-1 task root already exists" in text
    assert "segment-1 runtime closure already exists" in text
    assert "pending_external_sacct_terminal_finalization" in text
