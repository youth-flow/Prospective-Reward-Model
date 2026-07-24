from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from smart_reward.config import config_hash, load_config
from smart_reward.phase2_config import load_phase2_config_bundle

ROOT = Path(__file__).parents[1]
BASE_DIGEST = "sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3"


def _locked_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in (
        (ROOT / "containers" / "requirements-hpc4.lock").read_text(encoding="utf-8").splitlines()
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", line), line
        name, version = line.split("==", maxsplit=1)
        key = name.lower().replace("_", "-")
        assert key not in versions
        versions[key] = version
    return versions


def test_hpc4_lock_has_exact_research_critical_versions() -> None:
    versions = _locked_versions()

    assert versions["transformers"] == "4.52.3"
    assert versions["peft"] == "0.15.2"
    assert versions["accelerate"] == "1.7.0"
    assert versions["datasets"] == "3.6.0"
    assert versions["huggingface-hub"] == "0.31.4"
    assert versions["tokenizers"] == "0.21.1"
    assert versions["numpy"] == "1.26.4"
    assert versions["pyarrow"] == "17.0.0"
    assert versions["fsspec"] == "2025.3.0"
    assert "torch" not in versions


def test_definition_locks_base_and_forbids_dependency_resolution() -> None:
    definition = (ROOT / "containers" / "prorm-hpc4.def").read_text(encoding="utf-8")

    assert BASE_DIGEST in definition
    assert f"From: docker.io/pytorch/pytorch@{BASE_DIGEST}" in definition
    assert "--only-binary=:all: --no-deps" in definition
    assert "--no-deps --no-build-isolation /opt/prorm" in definition
    assert "python -m pip check" in definition


def test_image_workflow_publishes_raw_sif_and_checks_public_access() -> None:
    workflow = (ROOT / ".github" / "workflows" / "build-hpc4-image.yml").read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n  packages: write" in workflow
    assert "apptainer build \\" in workflow
    assert "--reproducible" in workflow
    assert "apptainer push --authfile" in workflow
    assert "oras://${GHCR_PACKAGE}:${tag}" in workflow
    assert "Prove anonymous GHCR pull authorization" in workflow
    assert "sif_sha256" in workflow


def test_tracked_config_identities_match_exact_bytes_and_semantics() -> None:
    identity_path = ROOT / "configs" / "identities.json"
    payload = json.loads(identity_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "prorm-config-identities/v1"
    assert set(payload["configs"]) == {
        "configs/common_beta_pilot.yaml",
        "configs/common_beta_pilot_base.yaml",
        "configs/main.yaml",
        "configs/smoke.yaml",
    }
    for relative, entry in payload["configs"].items():
        path = ROOT / relative
        if relative == "configs/common_beta_pilot.yaml":
            bundle = load_phase2_config_bundle(path)
            config = bundle.config
            semantic_hash = bundle.design_identity
        else:
            config = load_config(path)
            semantic_hash = config_hash(config)
        run = config["run"]
        expected_seed_count = 1 if "seed" in run else len(run["seeds"])
        assert entry == {
            "config_hash": semantic_hash,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed_count": expected_seed_count,
        }


def test_hpc4_staging_and_submission_do_not_execute_images_on_login() -> None:
    stage_submit = (ROOT / "scripts" / "hpc4" / "submit_hf_stage.sh").read_text(encoding="utf-8")
    controlled_submit = (ROOT / "scripts" / "hpc4" / "submit_controlled.sh").read_text(
        encoding="utf-8"
    )
    stage_job = (ROOT / "scripts" / "hpc4" / "hf_stage.sbatch").read_text(encoding="utf-8")
    controlled_job = (ROOT / "scripts" / "hpc4" / "controlled.sbatch").read_text(encoding="utf-8")

    assert "apptainer exec" not in stage_submit
    assert "apptainer exec" not in controlled_submit
    for submit_script in (stage_submit, controlled_submit):
        assert 'identity_relative="configs/identities.json"' in submit_script
        assert "python3 -I -S -" in submit_script
        assert '"cat-file", "blob"' in submit_script
        assert '--export="ALL,' not in submit_script
    assert "apptainer exec --cleanenv" in stage_job
    assert "--no-mount home,cwd" in stage_job
    assert "PRORM_CONFIG_HASH" in stage_job
    assert "stage result inventory_sha256 is invalid" in stage_job
    assert "another Hugging Face staging job is writing the shared cache" in stage_job
    assert "--no-mount home,cwd" in controlled_job
    assert "PRORM_IMAGE escaped PRORM_PROJECT_ROOT" in controlled_job
    assert "PRORM_HF_CACHE escaped PRORM_PROJECT_ROOT" in controlled_job
    assert "[<index>|<start>-<end>]" in controlled_submit
    assert 'array_selection="${4:-}"' in controlled_submit
    assert (
        "array selection must be one index or one contiguous start-end range" in controlled_submit
    )
    assert "array selection exceeds configured seed indices" in controlled_submit
    assert "array selection index exceeds safe integer limit" in controlled_submit
    assert "max_safe_array_integer=2147483647" in controlled_submit
    assert 'array_spec="${array_start}-${array_end}%${concurrency}"' in controlled_submit
    assert '--array="${array_spec}"' in controlled_submit


def test_controlled_submit_rejects_wrapping_array_index_before_arithmetic() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is unavailable on this host")

    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(("PRORM_", "SRM_", "APPTAINER", "SINGULARITY")):
            environment.pop(name)
    result = subprocess.run(
        [
            bash,
            str(ROOT / "scripts" / "hpc4" / "submit_controlled.sh"),
            "configs/main.yaml",
            "gpu-l20",
            "12:00:00",
            "18446744073709551620",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == (
        "error: array selection index exceeds safe integer limit 2147483647"
    )


def test_hpc4_formal_aggregation_is_cpu_slurm_and_commit_bound() -> None:
    submit = (ROOT / "scripts" / "hpc4" / "submit_aggregate.sh").read_text(encoding="utf-8")
    job = (ROOT / "scripts" / "hpc4" / "aggregate.sbatch").read_text(encoding="utf-8")

    # The login node only validates committed byte identities and submits a
    # CPU job. It must never execute the research image or guess a retry.
    assert "apptainer exec" not in submit
    assert "[--source-commit <full_commit>] <seed=controlled_job_id>..." in submit
    assert 'seed_job_pairs=("$@")' in submit
    assert "latest" not in submit.lower()
    assert 'identity_relative="configs/identities.json"' in submit
    assert "python3 -I -S -" in submit
    assert '"cat-file", "blob"' in submit
    assert 'source_commit="${source_commit_input:-${control_commit}}"' in submit
    assert 'merge-base --is-ancestor \\\n  "${source_commit}" "${control_commit}"' in submit
    assert '"${repo_root}" "${source_commit}" "${identity_relative}"' in submit
    assert '"${source_commit}:${config_relative}"' in submit
    assert "worktree config bytes do not match the source Git commit" in submit
    assert "committed config bytes do not match the committed identity" in submit
    assert "PRORM_CONTROL_COMMIT=${control_commit}" in submit
    assert "PRORM_SOURCE_COMMIT=${source_commit}" in submit
    assert "duplicate seed mapping" in submit
    assert "controlled job ID is mapped more than once" in submit
    assert "SBATCH_*" in submit
    assert "formal submission forbids ambient sbatch option overrides" in submit
    assert '--partition="${partition}"' in submit
    assert "--cpus-per-task=1" in submit
    assert "--mem=8G" in submit
    assert '--export="PATH=' in submit
    assert '--export="ALL,' not in submit
    assert '"${seed_job_pairs[@]}"' in submit

    # Aggregation itself is a zero-GPU CPU Slurm workload with an independent
    # exact-commit checkout and an isolated container mount namespace.
    assert "#SBATCH --account=sigroup" in job
    assert "#SBATCH --cpus-per-task=1" in job
    assert "#SBATCH --mem=8G" in job
    assert "#SBATCH --gpus" not in job
    assert "amd|intel)" in job
    assert "formal aggregation must not request GPUs" in job
    assert "formal aggregation must be a single non-array CPU job" in job
    assert "SLURM_JOB_GPUS" in job
    assert "CUDA_VISIBLE_DEVICES" in job
    assert "NVIDIA_VISIBLE_DEVICES" in job
    assert "SLURM_TRES_PER_NODE" in job
    assert "apptainer exec --cleanenv" in job
    assert "--no-mount home,cwd,bind-paths" in job
    assert "--nv" not in job
    assert '--pwd "${execution_repo}"' in job
    assert '--bind "${job_dir}:${job_dir},${PRORM_PROJECT_ROOT}:${PRORM_PROJECT_ROOT}"' in job
    assert "git clone --quiet --no-hardlinks --no-checkout" in job
    assert 'repo_head}" = "${PRORM_CONTROL_COMMIT}"' in job
    assert (
        'merge-base --is-ancestor \\\n  "${PRORM_SOURCE_COMMIT}" "${PRORM_CONTROL_COMMIT}"' in job
    )
    assert 'checkout --quiet --detach "${PRORM_SOURCE_COMMIT}"' in job
    assert "detached committed config identity differs from submission" in job
    assert "container-computed config hash differs from submitted identity" in job
    assert '"schema_version": "prorm-aggregation-sources/v2"' in job
    assert '"control_plane_git_commit": expected_control_commit' in job
    assert '"source_git_commit": expected_source_commit' in job
    assert '"schema_version": "prorm-aggregation-manifest/v2"' in job
    assert '"control_plane": {' in job
    assert '"aggregation_source": {' in job


def test_hpc4_formal_aggregation_revalidates_sources_and_publishes_atomically() -> None:
    job = (ROOT / "scripts" / "hpc4" / "aggregate.sbatch").read_text(encoding="utf-8")

    # Shell/Python preflight validates the explicit run paths, completion
    # marker, non-symlink inputs, and exact content-addressed artifact.
    assert 'f"job-{controlled_job}"' in job
    assert 'success_path = run_dir / "SUCCESS"' in job
    assert "parse_success(success_path, controlled_job)" in job
    assert 'slurm.get("SLURM_JOB_ID") != str(expected_job)' in job
    assert 'os.path.lexists(run_dir / "FAILED")' in job
    for name in (
        "comparison.json",
        "rollout.json",
        "run-manifest.json",
        "updated_rollouts.jsonl",
    ):
        assert f'"{name}"' in job
    assert "must be a regular non-symlink file" in job
    assert "run artifact must be a relative symlink" in job
    assert "run artifact symlink resolves to the wrong content address" in job
    assert '"artifact_symlink_target": link_target' in job
    assert '"success_sha256": success_sha' in job
    assert 'comparison.get("artifact_metadata_sha256") != artifact_metadata_sha' in job
    assert "artifact producer identity is invalid" in job

    # Source validation runs on both sides of aggregate-results. The CLI owns
    # semantic validation (all seeds, manifests, KL convergence, and metrics).
    assert job.count('validate_sources "${source_records_') == 3
    assert 'cmp -s -- "${source_records_before}" "${source_records_after}"' in job
    assert 'cmp -s -- "${source_records_before}" "${source_records_final}"' in job
    assert "python -m smart_reward.cli aggregate-results" in job
    assert '--repo-root "${execution_repo}"' in job
    assert '--rollouts "${rollouts[@]}"' in job
    assert "aggregate source paths or SHA256 identities are not exact" in job
    assert "image changed before atomic aggregation publication" in job
    assert "HF inventory changed before atomic aggregation publication" in job
    prepublish_snapshot = job.index('validate_sources "${source_records_final}"')
    prepublish_checkout = job.index("assert_execution_checkout", prepublish_snapshot)
    atomic_publish = job.index(
        'mv -T --no-clobber -- "${staging_dir}" "${aggregate_final}"',
        prepublish_checkout,
    )
    assert prepublish_snapshot < prepublish_checkout < atomic_publish
    assert '"sources": records["sources"]' in job

    # The staging directory is a sibling of aggregate/, is protected by a
    # campaign lock, and can only be installed with no-overwrite rename.
    assert 'exec {aggregate_lock_fd}> "${aggregate_lock}"' in job
    assert 'flock -n "${aggregate_lock_fd}"' in job
    assert 'mktemp -d "${campaign_root}/.aggregate.publish-${SLURM_JOB_ID}.XXXXXX"' in job
    assert "aggregation staging directory is not a sibling of the final directory" in job
    assert 'mv -T --no-clobber -- "${staging_dir}" "${aggregate_final}"' in job
    assert '[[ ! -e "${staging_dir}" && ! -L "${staging_dir}" ]]' in job
    assert "refusing to overwrite an existing formal aggregate" in job
    for name in (
        "aggregate.json",
        "aggregate.json.sha256",
        "aggregation-manifest.json",
        "SUCCESS",
    ):
        assert f'"{name}"' in job
    assert '"passed",\n    "not_passed",' in job
    assert "pre_registered_evidence_status" in job
