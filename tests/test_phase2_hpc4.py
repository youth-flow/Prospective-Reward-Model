from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_pilot.sh"
JOB = ROOT / "scripts" / "hpc4" / "phase2_pilot.sbatch"
AGGREGATE_SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_pilot_aggregate.sh"
AGGREGATE_JOB = ROOT / "scripts" / "hpc4" / "phase2_pilot_aggregate.sbatch"


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


def test_phase2_pilot_shell_sources_are_parseable_when_bash_is_available() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("Bash is unavailable on this host")
    for path in (SUBMIT, JOB, AGGREGATE_SUBMIT, AGGREGATE_JOB):
        result = subprocess.run(
            [bash, "-n", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_phase2_submit_is_login_safe_and_validates_both_committed_identities() -> None:
    submit = _text(SUBMIT)

    assert "apptainer exec" not in submit
    assert "<overlay.yaml> <base.yaml> <gpu-partition> <walltime>" in submit
    assert 'overlay_relative}" = "configs/common_beta_pilot.yaml"' in submit
    assert 'base_relative}" = "configs/common_beta_pilot_base.yaml"' in submit
    assert 'identity_relative="configs/identities.json"' in submit
    assert "python3 -I -S -" in submit
    assert '"cat-file", "blob"' in submit
    assert "worktree overlay bytes do not match submitted Git commit" in submit
    assert "worktree base config bytes do not match submitted Git commit" in submit
    assert "committed config bytes do not match committed identity" in submit
    assert "invalid semantic config identity" in submit
    assert "overlay source_config_hash does not equal the base semantic identity" in submit
    assert "Phase-2 pilot submission requires a clean Git worktree" in submit
    assert "Git worktree changed while preparing the Phase-2 submission" in submit
    assert '--export="ALL,' not in submit
    assert "submission forbids ambient sbatch option overrides" in submit
    assert "submission forbids ambient container controls" in submit


def test_phase2_submit_uses_base_inventory_and_two_way_array_concurrency() -> None:
    submit = _text(SUBMIT)

    assert "gpu-l20) ;;" in submit
    assert "Phase-2 design is locked to HPC4 gpu-l20" in submit
    for forbidden_partition in ("gpu-a30", "gpu-rtx5880", "gpu-rtx4090d"):
        assert forbidden_partition not in submit
    assert 'inventory_expected="${hf_cache}/inventories/${base_config_hash}.json"' in submit
    assert "missing base-config HF inventory" in submit
    assert "concurrency=2" in submit
    assert '[[ "${concurrency}" =~ ^[12]$ ]]' in submit
    assert 'array_spec="${array_start}-${array_end}%${concurrency}"' in submit
    assert '--array="${array_spec}"' in submit
    assert "--account=sigroup" in submit
    assert "--gpus-per-node=1" in submit
    assert "--cpus-per-task=8" in submit
    assert "--mem=64G" in submit
    for identity in (
        "PRORM_PHASE2_DESIGN_SHA256",
        "PRORM_PHASE2_BASE_CONFIG_HASH",
        "PRORM_PHASE2_OVERLAY_FILE_SHA256",
        "PRORM_PHASE2_BASE_FILE_SHA256",
        "PRORM_IDENTITIES_FILE_SHA256",
        "PRORM_HF_INVENTORY_SHA256",
        "PRORM_GIT_COMMIT",
    ):
        assert f"{identity}=" in submit


def test_phase2_submit_identity_binds_optional_freeze_and_horizon_aggregates() -> None:
    submit = _text(SUBMIT)

    assert "--beta-source-aggregate" in submit
    assert "--horizon-parent-aggregate" in submit
    assert 'resolve_project_path "${beta_source_aggregate_input}" file' in submit
    assert 'resolve_project_path "${horizon_parent_aggregate_input}" file' in submit
    assert "beta_source_aggregate_sha256" in submit
    assert "horizon_parent_aggregate_sha256" in submit
    assert "beta-source aggregate changed during submission" in submit
    assert "horizon-parent aggregate changed during submission" in submit
    assert "PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=" in submit
    assert "PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=" in submit
    assert "PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=" in submit
    assert "PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=" in submit
    assert '"${value}" != *"="*' in submit


def test_phase2_job_is_fail_closed_offline_and_detached() -> None:
    job = _text(JOB)

    assert "#SBATCH --account=sigroup" in job
    assert "#SBATCH --gpus-per-node=1" in job
    assert '[[ "${SLURM_JOB_ACCOUNT:-}" = "sigroup" ]]' in job
    assert "gpu-l20) ;;" in job
    assert "Phase-2 design is locked to HPC4 gpu-l20" in job
    for forbidden_partition in ("gpu-a30", "gpu-rtx5880", "gpu-rtx4090d"):
        assert forbidden_partition not in job
    assert "Phase-2 pilot requires exactly one Slurm GPU on the node" in job
    assert "CUDA_VISIBLE_DEVICES must identify exactly one allocated GPU" in job
    assert "torch.cuda.device_count() != 1" in job
    assert 'partition != "gpu-l20" or "l20" not in name.lower()' in job
    assert "Phase-2 design requires gpu-l20 backed by an NVIDIA L20" in job
    assert "apptainer exec --cleanenv --nv" in job
    assert "--no-mount home,cwd,bind-paths" in job
    assert '--env "HF_HUB_OFFLINE=1"' in job
    assert '--env "TRANSFORMERS_OFFLINE=1"' in job
    assert '--env "HF_DATASETS_OFFLINE=1"' in job
    assert "--allow-download" not in job
    assert "git clone --quiet --no-hardlinks --no-checkout" in job
    assert 'checkout --quiet --detach "${PRORM_GIT_COMMIT}"' in job
    assert "detached execution checkout became dirty" in job


def test_phase2_job_rechecks_binds_and_forwards_optional_aggregate_inputs() -> None:
    job = _text(JOB)

    assert "validate_optional_aggregate()" in job
    assert '"beta-source aggregate"' in job
    assert '"horizon-parent aggregate"' in job
    assert '"${label} SHA256 mismatch on compute node"' in job
    assert (
        '--bind "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}:${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}"'
    ) in job
    assert (
        '--bind "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}:${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}"'
    ) in job
    assert "phase2_aggregate_flags+=(" in job
    assert '--beta-source-aggregate "${PRORM_PHASE2_BETA_SOURCE_AGGREGATE}"' in job
    assert '--horizon-parent-aggregate "${PRORM_PHASE2_HORIZON_PARENT_AGGREGATE}"' in job
    assert '"${phase2_aggregate_flags[@]}"' in job
    assert "beta-source aggregate changed during Phase-2 execution" in job
    assert "horizon-parent aggregate changed during Phase-2 execution" in job
    assert "beta_source_aggregate_sha256=%s" in job
    assert "horizon_parent_aggregate_sha256=%s" in job


def test_phase2_job_keeps_artifact_and_run_identities_separate() -> None:
    job = _text(JOB)

    artifact_address = (
        '"${PRORM_PROJECT_ROOT}/artifacts/${PRORM_PHASE2_BASE_CONFIG_HASH}/'
        "${PRORM_IMAGE_SHA256}/${PRORM_HF_INVENTORY_SHA256}/"
        '${PRORM_GIT_COMMIT}/seed-${seed}"'
    )
    run_address = (
        '"${PRORM_PROJECT_ROOT}/runs/phase2-pilot/'
        '${PRORM_PHASE2_DESIGN_SHA256}/seed-${seed}/job-${execution_id}"'
    )
    assert artifact_address in job
    assert run_address in job
    assert "inventories/${PRORM_PHASE2_BASE_CONFIG_HASH}.json" in job
    assert "phase2-config-check" in job
    assert "container-computed Phase-2 semantic identities are invalid" in job
    assert "container-computed base semantic identity is invalid" in job
    assert "controlled-materialize" in job
    assert '"${execution_base}" "${artifact_dir}" --seed "${seed}" --device cuda' in job
    assert "prepare_phase2_inputs(" in job
    phase2_command = job.index("python -m smart_reward.cli phase2-run")
    explicit_verification = job.index("inputs = prepare_phase2_inputs(")
    assert explicit_verification < phase2_command
    assert '"${execution_overlay}" "${artifact_dir}" "${manifest}" "${phase2_result}"' in job
    assert "phase2-pilot-diagnostics.diagnostics.jsonl" in job


def test_phase2_job_persists_evidence_and_atomic_terminal_marker() -> None:
    job = _text(JOB)

    for filename in (
        "phase2-config-check.json",
        "base-config-check.json",
        "hf-inventory-verification.json",
        "gpu-check.log",
        "run-manifest.json",
        "materialize.log",
        "memory-materialize.json",
        "artifact-materialization.json",
        "artifact-verification.json",
        "phase2-run.log",
        "memory-phase2.json",
        "phase2-failure-evidence.json",
        "phase2-pilot-diagnostics.json",
        "phase2-pilot-diagnostics.diagnostics.jsonl",
        "phase2-output-verification.json",
    ):
        assert filename in job
    assert "atomic_sync_file()" in job
    assert '"PRORM_FAILURE_EVIDENCE=${job_dir}/phase2-failure-evidence.json"' in job
    assert "atomic_write_marker()" in job
    assert "schema_version=prorm-phase2-run-status/v1" in job
    assert 'if (( final_exit == 0 )); then marker="SUCCESS"; else marker="FAILED"; fi' in job
    assert "mv -T --no-clobber" in job
    assert 'ln -s -- "${artifact_relative}" "${artifact_link}"' in job
    assert "diagnostics_sha256" in job
    assert "common-beta-pilot-diagnostics/v2" in job
    assert "common-beta-pilot-diagnostic-row/v2" in job
    assert "forbidden_keys" in job
    assert "final_oracle_session_opened" in job
    assert "pilot-measured-kl-gate/v1" in job
    assert "phase2-pre-oracle-safety-gate/v1" in job
    assert '"prompt_mean_p95_kl_cap": 0.02' in job
    assert '"prompt_mean_p99_kl_cap": 0.05' in job
    assert '"prompt_mean_maximum_kl_cap": 0.10' in job
    assert '"per_sequence_maximum_kl_cap": 0.20' in job
    assert '"reached_max_length_rate_cap": 0.05' in job
    assert "common-beta-design/v4" in job
    assert "submitted_beta_source_sha" in job
    assert "submitted_horizon_parent_sha" in job
    assert (
        'runtime_contract.get("beta_source_aggregate_sha256") != expected_beta_source_sha'
    ) in job
    assert "horizon_parent != expected_horizon_parent_sha" in job
    assert (
        'beta_evidence.get("beta_source_aggregate_sha256")\n        != expected_beta_source_sha'
    ) in job
    assert 'beta_evidence.get("beta_source_aggregate_sha256") != horizon_parent' not in job
    assert "allowed_horizon_sequence" in job
    assert "parent_pilot_aggregate_sha256" in job
    assert "publish_target_free_diagnostics_without_final_oracle" in job
    assert '"kl_gate_passed": passed' in job
    assert '"kl_measure_only": True' in job


def test_phase2_pilot_aggregate_submit_is_static_cpu_control_plane() -> None:
    submit = _text(AGGREGATE_SUBMIT)

    assert "<run-dir-1> <run-dir-2> <run-dir-3>" in submit
    assert "apptainer exec" not in submit
    assert "pilot aggregation partition must be amd or intel" in submit
    assert "pilot aggregate submission requires a clean committed worktree" in submit
    assert "submission forbids container controls" in submit
    assert "submission forbids sbatch overrides" in submit
    assert "configs/common_beta_pilot.yaml" in submit
    assert "configs/common_beta_pilot_base.yaml" in submit
    assert "configs/identities.json" in submit
    assert "pilot aggregate requires exactly three seeds" in submit
    assert "refusing to overwrite pilot aggregate" in submit
    assert "status=SUCCESS" in submit
    assert "phase2_design_sha256=${design_sha}" in submit
    assert "base_config_hash=${base_hash}" in submit
    assert "git_commit=${producer_git_commit}" in submit
    assert "--producer-commit" in submit
    assert "producer commit must be an ancestor of the aggregation commit" in submit
    assert "producer and aggregator commits do not bind identical input" in submit
    assert "phase2-pilot-diagnostics.json" in submit
    assert "phase2-pilot-diagnostics.diagnostics.jsonl" in submit
    assert "run-manifest.json" in submit
    assert "phase2-output-verification.json" in submit
    assert "artifact/metadata.json" in submit
    assert "pilot result changed before submission" in submit
    assert "pilot sidecar changed before submission" in submit
    assert "pilot SUCCESS marker changed before submission" in submit
    assert "pilot run manifest changed before submission" in submit
    assert "pilot output verification changed before submission" in submit
    assert "pilot artifact metadata changed before submission" in submit
    assert "--beta-source-aggregate" in submit
    assert "--horizon-parent-aggregate" in submit
    assert "beta-source aggregate changed before submission" in submit
    assert "horizon-parent aggregate changed before submission" in submit
    assert "PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=" in submit
    assert "PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=" in submit
    assert "PRORM_PHASE2_DESIGN_SHA256=" in submit
    assert "PRORM_PHASE2_BASE_CONFIG_HASH=" in submit
    assert "PRORM_PHASE2_AGGREGATOR_GIT_COMMIT=" in submit
    assert "PRORM_PHASE2_PRODUCER_GIT_COMMIT=" in submit
    assert "PRORM_PHASE2_AGGREGATE_VALIDATOR_SOURCE_SHA256=" in submit
    assert "PRORM_IMAGE_SHA256=" in submit
    assert "PRORM_HF_INVENTORY_SHA256=" in submit
    assert "PRORM_PHASE2_RESULT_SHA256_${index}=" in submit
    assert "PRORM_PHASE2_SIDECAR_SHA256_${index}=" in submit
    assert "PRORM_PHASE2_SUCCESS_SHA256_${index}=" in submit
    assert "PRORM_PHASE2_MANIFEST_SHA256_${index}=" in submit
    assert "PRORM_PHASE2_OUTPUT_VERIFICATION_SHA256_${index}=" in submit
    assert "PRORM_PHASE2_ARTIFACT_METADATA_SHA256_${index}=" in submit
    assert "--account=sigroup" in submit
    assert '--partition="${partition}"' in submit
    assert "--gpus-per-node" not in submit
    assert "sbatch \\" in submit
    assert '"${repo_root}/scripts/hpc4/phase2_pilot_aggregate.sbatch"' in submit


def test_phase2_pilot_aggregate_job_is_exact_three_detached_and_atomic() -> None:
    job = _text(AGGREGATE_JOB)

    assert "#SBATCH --account=sigroup" in job
    assert ': "${SLURM_JOB_ID:?pilot aggregation must execute inside Slurm}"' in job
    assert "amd|intel" in job
    assert "pilot aggregation must be a single non-array job" in job
    assert "pilot aggregation must not receive GPU visibility" in job
    assert "pilot aggregation must not receive GPU GRES" in job
    assert "--nv" not in job
    assert "PRORM_PHASE2_RUN_DIR_0" in job
    assert "PRORM_PHASE2_RUN_DIR_1" in job
    assert "PRORM_PHASE2_RUN_DIR_2" in job
    assert "pilot aggregate inputs must be three distinct SUCCESS runs" in job
    assert "status=SUCCESS" in job
    assert "phase2_design_sha256=${PRORM_PHASE2_DESIGN_SHA256}" in job
    assert "base_config_hash=${PRORM_PHASE2_BASE_CONFIG_HASH}" in job
    assert "git_commit=${PRORM_PHASE2_PRODUCER_GIT_COMMIT}" in job
    assert "pilot result SHA256 mismatch" in job
    assert "pilot sidecar SHA256 mismatch" in job
    assert "pilot SUCCESS marker SHA256 mismatch" in job
    assert "pilot run manifest SHA256 mismatch" in job
    assert "pilot output verification SHA256 mismatch" in job
    assert "pilot artifact metadata SHA256 mismatch" in job
    assert "git clone --quiet --no-hardlinks --no-checkout" in job
    assert '"${PRORM_PHASE2_AGGREGATOR_GIT_COMMIT}"' in job
    assert "--producer-git-commit" in job
    assert "--aggregator-git-commit" in job
    assert "--validator-source-sha256" in job
    assert "apptainer exec --cleanenv" in job
    assert "--no-mount home,cwd,bind-paths" in job
    assert "phase2-config-check" in job
    assert "config-check" in job
    assert "phase2-pilot-aggregate" in job
    assert '"${results[@]}"' in job
    assert '"${aggregate_flags[@]}"' in job
    assert "--beta-source-aggregate" in job
    assert "--horizon-parent-aggregate" in job
    assert "common-beta-pilot-selection-aggregate/v2" in job
    assert "phase2-pilot-aggregation-identity/v1" in job
    assert "oracle_outcomes_consumed" in job
    assert "formal_efficacy_evidence_produced" in job
    assert "pilot aggregate input changed during execution" in job
    assert "image changed during pilot aggregation" in job
    assert "inventory changed during pilot aggregation" in job
    assert "detached pilot aggregate checkout changed" in job
    assert "control-plane checkout changed during pilot aggregation" in job
    assert "staged_output_sha256=" in job
    assert 'mv -T --no-clobber -- "${staged_output}" "${output}"' in job
    staged_sync = job.index('fsync_file_and_parent "${staged_output}"')
    atomic_publish = job.index('mv -T --no-clobber -- "${staged_output}" "${output}"')
    canonical_sync = job.index('fsync_file_and_parent "${output}"')
    publication_log = job.index("Phase-2 pilot aggregate published:")
    assert staged_sync < atomic_publish < canonical_sync < publication_log
    assert '[[ ! -e "${staged_output}" && ! -L "${staged_output}" ]]' in job
    assert '[[ -f "${output}" && ! -L "${output}" ]]' in job
    assert 'printf \'%s  %s\\n\' "${staged_output_sha256}" "${output}"' in job
    assert "destination won a publication race; staged file was not installed" in job
    assert "atomic publication did not install a regular pilot aggregate" in job
    assert "published pilot aggregate differs from the staged payload" in job
