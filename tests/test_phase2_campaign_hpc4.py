from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SUBMIT = ROOT / "scripts" / "hpc4" / "submit_phase2_campaign_finalize.sh"
JOB = ROOT / "scripts" / "hpc4" / "phase2_campaign_finalize.sbatch"


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


def _shell_function_source(source: str, name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


def test_phase2_campaign_shell_sources_are_parseable_when_bash_is_available() -> None:
    for path in (SUBMIT, JOB):
        programs = _embedded_python_sources(_text(path))
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


def test_campaign_submit_binds_clean_exact_confirmatory_source_and_inventory() -> None:
    submit = _text(SUBMIT)

    assert "campaign finalization submission requires a clean committed worktree" in submit
    assert "worktree bytes differ from committed exact source" in submit
    assert 'git -C "${repo_root}" cat-file blob' in submit
    assert 'identity_relative="configs/identities.json"' in submit
    assert 'value["seed_count"] != 30' in submit
    assert "formal campaign requires exactly 30 seeds" in submit
    assert "invalid semantic identity" in submit
    assert "confirmatory design and base identities must differ" in submit
    assert 'inventory_expected="${hf_cache}/inventories/${base_config_hash}.json"' in submit
    assert "HF inventory is unsafe or not addressed by the base identity" in submit
    assert "image SHA256 mismatch" in submit
    assert "inventory changed before submission" in submit
    assert "committed identity input changed before submission" in submit
    assert "apptainer exec" not in submit
    assert "submission forbids container controls" in submit
    assert "submission forbids sbatch overrides" in submit


def test_campaign_submit_requires_ordered_exact_30_terminal_manifests_and_markers() -> None:
    submit = _text(SUBMIT)

    assert "campaign-plan.json" in submit
    assert "PRORM_PHASE2_CAMPAIGN_PLAN_SHA256" in submit
    assert "expected_terminal_count=30" in submit
    assert "expected_argument_count=6" in submit
    assert "<30 terminal manifests" not in submit
    assert "resolve_phase2_campaign_registry.py" in submit
    assert "campaign registry did not resolve exactly 30 terminal heads" in submit
    assert "read_terminal_seed()" in submit
    assert "index=$((seed - 20260901))" in submit
    assert "duplicate terminal manifest for seed" in submit
    assert "seed=$((20260901 + index))" in submit
    assert "formal campaign requires exactly 30 terminal inputs" in submit
    assert '"${campaign_root}/seed-${seed}/"*' in submit
    assert "validate_phase2_terminal.py" in submit
    assert "prior_attempt_ledger_sha256" not in submit
    assert "retry_eligible" not in submit
    assert "infrastructure_failure_pre_outcome" not in submit
    assert "terminal input or marker changed before submission" in submit
    assert "PRORM_PHASE2_TERMINAL_COUNT=30" in submit
    assert "PRORM_PHASE2_TERMINAL_${slot}=" in submit
    assert "PRORM_PHASE2_TERMINAL_SHA256_${slot}=" in submit
    assert "PRORM_PHASE2_MARKER_${slot}=" in submit
    assert "PRORM_PHASE2_MARKER_SHA256_${slot}=" in submit


def test_campaign_submit_is_cpu_only_and_reserves_one_atomic_output_directory() -> None:
    submit = _text(SUBMIT)

    assert "campaign finalization partition must be amd or intel" in submit
    assert "--account=sigroup" in submit
    assert '--partition="${partition}"' in submit
    assert "--gpus-per-node" not in submit
    assert "terminal and aggregate outputs must share one new publication directory" in submit
    assert "one-shot campaign-final directory" in submit
    assert "campaign output filenames are locked" in submit
    assert "refusing to overwrite campaign publication directory" in submit
    assert "campaign publication destination appeared before submission" in submit
    assert '"${repo_root}/scripts/hpc4/phase2_campaign_finalize.sbatch"' in submit


def test_campaign_job_is_single_cpu_offline_and_detached() -> None:
    job = _text(JOB)

    assert "#SBATCH --account=sigroup" in job
    assert ': "${SLURM_JOB_ID:?campaign finalization must execute inside Slurm}"' in job
    assert "amd|intel" in job
    assert "campaign finalization must be a single non-array job" in job
    assert "campaign finalization must not receive GPU allocation or visibility" in job
    assert "campaign finalization must not receive GPU GRES/TRES" in job
    assert "--nv" not in job
    assert "git clone --quiet --no-hardlinks --no-checkout" in job
    assert 'checkout --quiet --detach "${PRORM_GIT_COMMIT}"' in job
    assert "detached campaign finalization checkout changed" in job
    assert "control-plane checkout changed during campaign finalization" in job
    assert "apptainer exec --cleanenv" in job
    assert "--no-mount home,cwd,bind-paths" in job
    assert '--env "HF_HUB_OFFLINE=1"' in job
    assert '--env "TRANSFORMERS_OFFLINE=1"' in job
    assert '--env "HF_DATASETS_OFFLINE=1"' in job


def test_campaign_job_revalidates_confirmatory_config_inventory_and_all_inputs() -> None:
    job = _text(JOB)

    assert '[[ "${PRORM_PHASE2_TERMINAL_COUNT}" = "30" ]]' in job
    assert "for index in {0..29}; do" in job
    assert "seed=$((20260901 + index))" in job
    assert "phase2-config-check" in job
    assert "config-check" in job
    assert 'phase2.get("design_stage") != "confirmatory"' in job
    assert 'phase2.get("formal_eligibility") is not True' in job
    assert "container confirmatory identities differ from submission" in job
    assert "confirmatory-contract-check.json" in job
    assert "detached confirmatory contract differs from terminal markers" in job
    assert "PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256" in job
    assert "stage_hf_assets.py" in job
    assert "--verify-only --inventory" in job
    assert "offline HF inventory verification changed the base identity" in job
    assert "terminal manifest SHA256 mismatch" in job
    assert "terminal marker SHA256 mismatch" in job
    assert "terminal input, marker, or nested result changed before finalization" in job
    assert "terminal input, marker, or nested result changed during finalization" in job
    assert "validate_phase2_terminal.py" in job
    assert "prior_attempt_ledger_sha256" not in job
    assert "retry_eligible" not in job
    assert "infrastructure_failure_pre_outcome" not in job
    assert job.count("resolve_phase2_campaign_registry.py") >= 2
    assert "submitted terminal is not the current registry head" in job
    assert "campaign_inputs_snapshot_sha256()" in job
    assert "campaign registry or attempt evidence changed during finalization" in job
    assert "campaign_inputs_snapshot_sha256" in job
    assert "failed to acquire exclusive campaign registry finalization lock" in job


def test_campaign_job_uses_terminal_finalizer_and_atomic_no_overwrite_publication() -> None:
    job = _text(JOB)

    assert "python -m smart_reward.cli phase2-campaign-finalize" in job
    assert "python -m smart_reward.cli phase2-aggregate" not in job
    assert '"${terminals[@]}"' in job
    assert "phase2-campaign-terminal/v2" in job
    assert "common-beta-seed-aggregate/v3" in job
    assert "failed campaign attempted to publish a primary aggregate" in job
    assert "failed campaign produced a staged primary aggregate" in job
    assert "successful campaign is missing its staged primary aggregate" in job
    assert 'staging_dir="$(mktemp -d "${campaign_root}/.campaign.publish-' in job
    assert '"${staging_dir}" "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}"' in job
    assert "mv -T --no-clobber" in job
    assert "atomic no-overwrite campaign directory publication failed" in job
    assert "phase2-campaign-publication-receipt/v2" in job
    assert '"campaign_plan_sha256": campaign_plan_sha' in job
    assert 'source_bindings=("${campaign_plan}:${PRORM_PHASE2_CAMPAIGN_PLAN_SHA256}")' in job
    assert '"publication_commit": "atomic_directory_rename"' in job
    assert "fsync_tree()" in job
    assert "campaign publication staging tree differs from its declaration" in job
    assert "stat.S_ISREG" in job
    assert 'getattr(os, "O_NOFOLLOW", 0)' in job
    assert "stream.flush()" in job
    assert "os.fsync(stream.fileno())" in job
    receipt = job.index('publication_receipt="${staging_dir}/')
    tree_fsync = job.index(
        'fsync_tree \\\n  "${staging_dir}"',
        receipt,
    )
    rename = job.index('"${staging_dir}" "${PRORM_PHASE2_CAMPAIGN_OUTPUT_DIR}"')
    parent_fsync = job.index('fsync_directory "${campaign_root}"', rename)
    clear_staging = job.index('staging_dir=""', rename)
    assert receipt < tree_fsync < rename < parent_fsync < clear_staging
    assert "sha256sum --check" not in job[rename:]
    assert "|| true" in job[rename:]


def test_campaign_publication_tree_fsync_rejects_undeclared_or_unsafe_members(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        return
    job = _text(JOB)
    function = _shell_function_source(job, "fsync_tree")
    program = _embedded_python_sources(function)[0]
    terminal_name = "phase2-campaign-terminal.json"
    aggregate_name = "phase2-primary-aggregate.json"
    receipt_name = "phase2-publication-receipt.json"
    for name in (terminal_name, aggregate_name, receipt_name):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    def run(mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-",
                str(tmp_path),
                mode,
                terminal_name,
                aggregate_name,
            ],
            input=program,
            capture_output=True,
            text=True,
            check=False,
        )

    assert run("success").returncode == 0
    extra = tmp_path / "undeclared.json"
    extra.write_text("{}\n", encoding="utf-8")
    assert run("success").returncode != 0
    extra.unlink()
    terminal = tmp_path / terminal_name
    terminal.unlink()
    terminal.mkdir()
    assert run("success").returncode != 0
